# Copyright (c) 2023-present Plane Software, Inc. and contributors
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

"""Questimus fork change (QUESTIMUS-30): move a work item to another project.

POST /api/workspaces/<slug>/projects/<project_id>/issues/<issue_id>/move/
body: {"target_project_id": "<uuid>"}

Moves the issue — and every descendant of its subtree — into the destination
project inside a single transaction guarded by advisory locks on BOTH the
source and the destination project (the same lock pattern Issue.save uses for
sequence allocation; both keys taken in sorted order to avoid deadlocks).
"""

# Python imports
import json

# Django imports
from django.contrib.postgres.aggregates import ArrayAgg
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError
from django.core.serializers.json import DjangoJSONEncoder
from django.db import connection, transaction
from django.db.models import Count, Max, OuterRef, Q, Subquery, UUIDField, Value
from django.db.models.functions import Coalesce
from django.utils import timezone

# Third Party imports
from rest_framework import status
from rest_framework.response import Response

# Module imports
from plane.app.permissions import ROLE, allow_permission
from plane.app.serializers import IssueSerializer
from plane.bgtasks.issue_activities_task import issue_activity
from plane.bgtasks.issue_version_sync import create_issue_version, get_related_data
from plane.bgtasks.webhook_task import model_activity
from plane.db.models import (
    CommentReaction,
    CycleIssue,
    Description,
    DescriptionVersion,
    FileAsset,
    GithubCommentSync,
    GithubIssueSync,
    IntakeIssue,
    Issue,
    IssueActivity,
    IssueAssignee,
    IssueBlocker,
    IssueComment,
    IssueDescriptionVersion,
    IssueLabel,
    IssueLink,
    IssueMention,
    IssueReaction,
    IssueRelation,
    IssueSequence,
    IssueSubscriber,
    IssueVote,
    IssueVersion,
    Label,
    ModuleIssue,
    Notification,
    Project,
    ProjectMember,
    State,
    UserFavorite,
    UserRecentVisit,
)
# Questimus fork change (QUESTIMUS-30): IssueAttachment is a legacy model not
# re-exported from plane.db.models (attachments live in FileAsset here), so
# import it from its defining module.
from plane.db.models.issue import IssueAttachment
from plane.db.models.issue_type import ProjectIssueType
from plane.utils.host import base_host
from plane.utils.uuid import convert_uuid_to_integer

from .. import BaseAPIView


def _collect_descendant_ids(root_id):
    """Return [root, ...descendants] with parents before their children.

    Soft-deleted children are included: they belong to the tree and must move
    with it (a soft-deleted child left behind would keep a parent_id pointing
    into the destination project). Their rows are simply re-pointed like any
    other moved row.
    """
    moving = [root_id]
    seen = {root_id}
    frontier = [root_id]
    while frontier:
        children = Issue.all_objects.filter(parent_id__in=frontier).values_list("id", flat=True)
        new = [child_id for child_id in children if child_id not in seen]
        moving.extend(new)
        seen.update(new)
        frontier = new
    return moving


def _annotate_issue_queryset(issues):
    """Add the queryset annotations IssueSerializer's read-only fields need.

    The serializer declares cycle_id/sub_issues_count/attachment_count/link_count
    (and label_ids/assignee_ids/module_ids) as fields sourced from instance
    attributes; without these annotations the fields are silently omitted from
    the response (DRF SkipField). Annotate so the move response matches the
    list/detail payload shape (mirrors IssueViewSet.apply_annotations and
    partial_update's ArrayAgg annotations).
    """
    return (
        issues.annotate(
            cycle_id=Subquery(
                CycleIssue.objects.filter(issue=OuterRef("id"), deleted_at__isnull=True).values("cycle_id")[:1]
            )
        )
        .annotate(
            link_count=Subquery(
                IssueLink.objects.filter(issue=OuterRef("id")).values("issue").annotate(count=Count("id")).values("count")
            )
        )
        .annotate(
            attachment_count=Subquery(
                FileAsset.objects.filter(issue_id=OuterRef("id"), entity_type=FileAsset.EntityTypeContext.ISSUE_ATTACHMENT)
                .values("issue_id")
                .annotate(count=Count("id"))
                .values("count")
            )
        )
        .annotate(
            sub_issues_count=Subquery(
                Issue.issue_objects.filter(parent=OuterRef("id")).values("parent").annotate(count=Count("id")).values("count")
            )
        )
        .annotate(
            label_ids=Coalesce(
                ArrayAgg(
                    "labels__id",
                    distinct=True,
                    filter=Q(~Q(labels__id__isnull=True) & Q(label_issue__deleted_at__isnull=True)),
                ),
                Value([], output_field=ArrayField(UUIDField())),
            )
        )
        .annotate(
            assignee_ids=Coalesce(
                ArrayAgg(
                    "assignees__id",
                    distinct=True,
                    filter=Q(
                        ~Q(assignees__id__isnull=True)
                        & Q(assignees__member_project__is_active=True)
                        & Q(issue_assignee__deleted_at__isnull=True)
                    ),
                ),
                Value([], output_field=ArrayField(UUIDField())),
            )
        )
        .annotate(
            module_ids=Coalesce(
                ArrayAgg(
                    "issue_module__module_id",
                    distinct=True,
                    filter=Q(
                        ~Q(issue_module__module_id__isnull=True)
                        & Q(issue_module__module__archived_at__isnull=True)
                        & Q(issue_module__deleted_at__isnull=True)
                    ),
                ),
                Value([], output_field=ArrayField(UUIDField())),
            )
        )
    )


def _remap_state(issue, target_project_id):
    """Resolve the destination state for a moved issue.

    Order: same-name state in the destination (case-insensitive) -> the
    destination default state -> the first non-triage destination state.
    Uses State.objects (excludes soft-deleted rows and triage states) so a
    soft-deleted source state can never be re-pointed to.
    """
    if issue.state_id is not None:
        same_name = State.objects.filter(
            project_id=target_project_id, name__iexact=issue.state.name
        ).first()
        if same_name is not None:
            return same_name
    return (
        State.objects.filter(project_id=target_project_id, default=True).first()
        or State.objects.filter(project_id=target_project_id).first()
    )


class IssueMoveEndpoint(BaseAPIView):
    """Move a work item (and its subtree) into another project of the workspace."""

    @allow_permission([ROLE.ADMIN, ROLE.MEMBER])
    def post(self, request, slug, project_id, issue_id):
        # --- Payload -------------------------------------------------------
        target_project_id = request.data.get("target_project_id", None)
        if not target_project_id:
            return Response(
                {"error": "target_project_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            target_project = Project.objects.get(pk=target_project_id)
        except (Project.DoesNotExist, ValidationError, ValueError):
            return Response(
                {"error": "target_project_id is not valid"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --- Source project -------------------------------------------------
        source_project = Project.objects.filter(pk=project_id, workspace__slug=slug).first()
        if source_project is None:
            return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)

        # Single-workspace fork: cross-workspace targets are not allowed.
        if target_project.workspace_id != source_project.workspace_id:
            return Response(
                {"error": "Cannot move issues across workspaces"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if target_project.id == source_project.id:
            return Response(
                {"error": "Target project must be different from the source project"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --- Destination membership (allow_permission covers only the source) -
        if not ProjectMember.objects.filter(
            project_id=target_project.id,
            member=request.user,
            role__gte=15,
            is_active=True,
        ).exists():
            return Response(
                {"error": "You don't have the required permissions."},
                status=status.HTTP_403_FORBIDDEN,
            )

        # --- Source issue ----------------------------------------------------
        issue = _annotate_issue_queryset(
            Issue.all_objects.filter(
                pk=issue_id, project_id=project_id, workspace__slug=slug, deleted_at__isnull=True
            )
        ).first()
        if issue is None:
            return Response({"error": "Issue not found"}, status=status.HTTP_404_NOT_FOUND)

        if issue.archived_at is not None:
            return Response(
                {"error": "Archived issues cannot be moved"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if IntakeIssue.objects.filter(issue_id=issue.id).exists():
            return Response(
                {"error": "Issues submitted through intake cannot be moved"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # The destination must have at least one non-triage state; otherwise the
        # state remap has nothing to resolve to and a moved issue would end up
        # state-less while keeping its completed_at (review REAL-6).
        if not State.objects.filter(project_id=target_project.id).exists():
            return Response(
                {"error": "Destination project does not have any states configured"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        current_instance = json.dumps(IssueSerializer(issue).data, cls=DjangoJSONEncoder)

        with transaction.atomic():
            # Destination-project advisory lock (same pattern as Issue.save):
            # only one move per destination project may allocate sequences at a
            # time. The source-project lock is taken as well (in sorted order to
            # avoid deadlocks) so two concurrent moves of the same issue to
            # different destinations cannot interleave (review REAL-5).
            with connection.cursor() as cursor:
                for lock_key in sorted(
                    [
                        convert_uuid_to_integer(source_project.id),
                        convert_uuid_to_integer(target_project.id),
                    ]
                ):
                    cursor.execute("SELECT pg_advisory_xact_lock(%s)", [lock_key])

            moving_ids = _collect_descendant_ids(issue.id)
            moved_issues = list(
                Issue.all_objects.select_related("state").filter(pk__in=moving_ids)
            )

            # Type availability for the destination is project-scoped; resolve
            # it once instead of per issue (review S3).
            dest_type_ids = set(
                ProjectIssueType.objects.filter(
                    project_id=target_project.id, deleted_at__isnull=True
                ).values_list("issue_type_id", flat=True)
            )
            dest_default_type_id = (
                ProjectIssueType.objects.filter(
                    project_id=target_project.id, is_default=True, deleted_at__isnull=True
                )
                .values_list("issue_type_id", flat=True)
                .first()
            )

            for moving in moved_issues:
                # 4. State remap (completed_at syncs via Issue.save's _sync_completed_at)
                new_state = _remap_state(moving, target_project.id)

                # 7. Type remap: only types linked to the destination project stay
                new_type_id = moving.type_id
                if new_type_id is not None and new_type_id not in dest_type_ids:
                    new_type_id = dest_default_type_id

                # 2. Sequence reassignment: max(dest) + 1
                new_sequence = (
                    IssueSequence.objects.filter(project_id=target_project.id).aggregate(
                        largest=Max("sequence")
                    )["largest"]
                    or 0
                ) + 1

                # 9. sort_order recomputed per destination project + state
                largest_sort_order = Issue.objects.filter(
                    project_id=target_project.id, state=new_state
                ).aggregate(largest=Max("sort_order"))["largest"]
                new_sort_order = largest_sort_order + 10000 if largest_sort_order is not None else 65535.0

                is_root = moving.pk == issue.pk
                moving.state = new_state
                moving.type_id = new_type_id
                moving.project_id = target_project.id
                moving.sequence_id = new_sequence
                moving.sort_order = new_sort_order
                moving.estimate_point_id = None
                update_fields = [
                    "project_id",
                    "sequence_id",
                    "state_id",
                    "type_id",
                    "sort_order",
                    "estimate_point",
                ]
                if is_root:
                    # 3. The subtree root detaches from its non-moving parent.
                    moving.parent_id = None
                    update_fields.append("parent_id")
                moving.save(update_fields=update_fields)

                # New destination sequence row; the old source row is kept.
                IssueSequence.objects.create(
                    issue=moving, sequence=new_sequence, project=target_project
                )

            root = next(m for m in moved_issues if m.pk == issue.pk)

            # 5. Labels: workspace-level labels are kept as-is. Project labels
            #    of the source project are ADOPTED into the destination (the
            #    Label row is re-pointed) so moved work items keep their labels
            #    (decision amendment 2026-09-11 on request); if the destination
            #    already has a same-named label, the bridge row is dropped
            #    instead to respect the (project, name) uniqueness. Labels
            #    already scoped to the destination stay untouched. Label
            #    hierarchy parents not attached to a moved issue stay in the
            #    source (cosmetic cross-project parent; acceptable).
            IssueLabel.objects.filter(issue_id__in=moving_ids).filter(
                label__project_id=source_project.id
            ).filter(
                label__name__in=Label.objects.filter(
                    project_id=target_project.id, deleted_at__isnull=True
                ).values("name")
            ).delete()
            source_label_ids = list(
                IssueLabel.objects.filter(issue_id__in=moving_ids)
                .filter(label__project_id=source_project.id)
                .values_list("label_id", flat=True)
            )
            if source_label_ids:
                # Destination labels sharing a name with a source label block
                # adoption: their bridge rows were already dropped above.
                colliding_label_ids = set(
                    Label.objects.filter(
                        project_id=target_project.id,
                        deleted_at__isnull=True,
                        name__in=Label.objects.filter(pk__in=source_label_ids).values("name"),
                    )
                    .exclude(pk__in=source_label_ids)
                    .values_list("id", flat=True)
                )
                adopted_label_ids = [i for i in source_label_ids if i not in colliding_label_ids]
                if adopted_label_ids:
                    Label.objects.filter(pk__in=adopted_label_ids).update(project_id=target_project.id)
            IssueLabel.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)

            # 6. Assignees: keep only active destination members (role >= 15).
            kept_assignee_ids = set(
                ProjectMember.objects.filter(
                    project_id=target_project.id, role__gte=15, is_active=True
                )
                .exclude(member_id__isnull=True)
                .values_list("member_id", flat=True)
            )
            IssueAssignee.objects.filter(issue_id__in=moving_ids).exclude(
                assignee_id__in=kept_assignee_ids
            ).delete()
            IssueAssignee.objects.filter(issue_id__in=moving_ids).update(
                project_id=target_project.id
            )

            # 8. Cycle / module bridges are project-scoped: drop them.
            CycleIssue.objects.filter(issue_id__in=moving_ids).delete()
            ModuleIssue.objects.filter(issue_id__in=moving_ids).delete()

            # 10. Relations/blockers survive only when both sides end up in the
            #     destination (co-moving or already there).
            dest_issue_ids = set(
                Issue.all_objects.filter(
                    project_id=target_project.id, deleted_at__isnull=True
                ).values_list("id", flat=True)
            )
            kept_ids = set(moving_ids) | dest_issue_ids
            relation_rows = IssueRelation.objects.filter(
                Q(issue_id__in=moving_ids) | Q(related_issue_id__in=moving_ids)
            )
            relation_rows.exclude(
                Q(issue_id__in=kept_ids) & Q(related_issue_id__in=kept_ids)
            ).delete()
            relation_rows.filter(
                Q(issue_id__in=kept_ids) & Q(related_issue_id__in=kept_ids)
            ).update(project_id=target_project.id)

            blocker_rows = IssueBlocker.objects.filter(
                Q(block_id__in=moving_ids) | Q(blocked_by_id__in=moving_ids)
            )
            blocker_rows.exclude(
                Q(block_id__in=kept_ids) & Q(blocked_by_id__in=kept_ids)
            ).delete()
            blocker_rows.filter(
                Q(block_id__in=kept_ids) & Q(blocked_by_id__in=kept_ids)
            ).update(project_id=target_project.id)

            # 10. Related rows follow the issue: bulk project_id update
            #     (workspace unchanged). IssueSequence is excluded on purpose:
            #     the old source rows are kept as historical records (item 2)
            #     and the new destination rows are already project-scoped.
            IssueActivity.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            IssueComment.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            IssueLink.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            IssueAttachment.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            IssueSubscriber.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            IssueReaction.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            IssueVote.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            IssueMention.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            CommentReaction.objects.filter(comment__issue_id__in=moving_ids).update(
                project_id=target_project.id
            )
            IssueVersion.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            IssueDescriptionVersion.objects.filter(issue_id__in=moving_ids).update(
                project_id=target_project.id
            )
            GithubIssueSync.objects.filter(issue_id__in=moving_ids).update(project_id=target_project.id)
            GithubCommentSync.objects.filter(comment__issue_id__in=moving_ids).update(
                project_id=target_project.id
            )

            # Comment description / description-versions (nullable project).
            comment_description_ids = IssueComment.objects.filter(
                issue_id__in=moving_ids
            ).values_list("description_id", flat=True)
            Description.objects.filter(pk__in=comment_description_ids).update(
                project_id=target_project.id
            )
            DescriptionVersion.objects.filter(description_id__in=comment_description_ids).update(
                project_id=target_project.id
            )

            # File assets bound to issues / their comments.
            FileAsset.objects.filter(
                Q(
                    entity_type=FileAsset.EntityTypeContext.ISSUE_ATTACHMENT,
                    issue_id__in=moving_ids,
                )
                | Q(
                    entity_type=FileAsset.EntityTypeContext.ISSUE_DESCRIPTION,
                    issue_id__in=moving_ids,
                )
                | Q(
                    entity_type=FileAsset.EntityTypeContext.COMMENT_DESCRIPTION,
                    comment__issue_id__in=moving_ids,
                )
            ).update(project_id=target_project.id)

            # Favorites / recent visits / notifications keyed by issue id.
            UserFavorite.objects.filter(entity_identifier__in=moving_ids).update(
                project_id=target_project.id
            )
            UserRecentVisit.objects.filter(
                entity_identifier__in=moving_ids, entity_name="issue"
            ).update(project_id=target_project.id)
            Notification.objects.filter(
                entity_identifier__in=moving_ids, entity_name="issue"
            ).update(project_id=target_project.id)

            # 11. Move activity row (created against the destination project).
            IssueActivity.objects.create(
                project_id=target_project.id,
                workspace_id=target_project.workspace_id,
                issue_id=root.id,
                verb="updated",
                field="project",
                old_value=str(source_project.id),
                new_value=str(target_project.id),
                actor=request.user,
                epoch=int(timezone.now().timestamp()),
            )

        # 11. Activity + webhook tasks (per IssueViewSet.partial_update).
        requested_data = json.dumps(request.data, cls=DjangoJSONEncoder)
        issue_activity.delay(
            type="issue.activity.updated",
            requested_data=requested_data,
            actor_id=str(request.user.id),
            issue_id=str(root.id),
            project_id=str(target_project.id),
            current_instance=current_instance,
            epoch=int(timezone.now().timestamp()),
            notification=True,
            origin=base_host(request=request, is_app=True),
        )
        model_activity.delay(
            model_name="issue",
            model_id=str(root.id),
            requested_data=request.data,
            current_instance=current_instance,
            actor_id=request.user.id,
            slug=slug,
            origin=base_host(request=request, is_app=True),
        )

        # 11. Version history: IssueVersion.log_issue_version is broken upstream
        #     (never passes project/workspace and silently swallows the error);
        #     create_issue_version only builds the row (the sync task bulk-creates
        #     it), so persist the constructed instance here (review REAL-2).
        _version = create_issue_version(root, get_related_data([root.id]))
        if _version is not None:
            _version.save()

        root = _annotate_issue_queryset(
            Issue.all_objects.filter(pk=root.id, deleted_at__isnull=True)
        ).first()
        serializer = IssueSerializer(root)
        return Response(serializer.data, status=status.HTTP_200_OK)

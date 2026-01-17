#!/bin/bash
# Ralph - Two-Agent Autonomous Coding System (Planner + Worker)
# Usage: ./ralph.sh [max_iterations]

set -e

MAX_ITERATIONS=${1:-10}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
PRD_FILE="$PROJECT_ROOT/prd.json"
SUBTASKS_FILE="$PROJECT_ROOT/subtasks.json"
PROGRESS_FILE="$PROJECT_ROOT/progress.txt"
ARCHIVE_DIR="$PROJECT_ROOT/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"

# Error tracking for planner failures
PLANNER_FAILURES=0
MAX_PLANNER_FAILURES=3

# --- Helper Functions ---

has_pending_subtasks() {
  if [ ! -f "$SUBTASKS_FILE" ]; then
    return 1  # No subtasks file = no pending subtasks
  fi

  # Check if any subtask has status: "pending"
  PENDING_COUNT=$(jq '[.subtasks[] | select(.status == "pending")] | length' "$SUBTASKS_FILE" 2>/dev/null || echo "0")
  [ "$PENDING_COUNT" -gt 0 ]
}

has_blocked_subtasks() {
  if [ ! -f "$SUBTASKS_FILE" ]; then
    return 1
  fi

  BLOCKED_COUNT=$(jq '[.subtasks[] | select(.status == "blocked")] | length' "$SUBTASKS_FILE" 2>/dev/null || echo "0")
  [ "$BLOCKED_COUNT" -gt 0 ]
}

all_subtasks_complete() {
  if [ ! -f "$SUBTASKS_FILE" ]; then
    return 1
  fi

  # All subtasks must have status: "completed"
  INCOMPLETE_COUNT=$(jq '[.subtasks[] | select(.status != "completed")] | length' "$SUBTASKS_FILE" 2>/dev/null || echo "1")
  [ "$INCOMPLETE_COUNT" -eq 0 ]
}

all_stories_complete() {
  if [ ! -f "$PRD_FILE" ]; then
    return 1
  fi

  # Check if all stories in prd.json have passes: true
  INCOMPLETE=$(jq '[.userStories[] | select(.passes == false)] | length' "$PRD_FILE" 2>/dev/null || echo "1")
  [ "$INCOMPLETE" -eq 0 ]
}

mark_story_complete() {
  if [ ! -f "$SUBTASKS_FILE" ]; then
    echo "ERROR: No subtasks.json to get story ID from"
    return 1
  fi

  # Get current story ID from subtasks.json
  STORY_ID=$(jq -r '.storyId' "$SUBTASKS_FILE")

  if [ -z "$STORY_ID" ] || [ "$STORY_ID" = "null" ]; then
    echo "ERROR: Could not get storyId from subtasks.json"
    return 1
  fi

  # Update prd.json to set passes: true for this story
  jq --arg id "$STORY_ID" \
    '(.userStories[] | select(.id == $id)).passes = true' \
    "$PRD_FILE" > "$PRD_FILE.tmp" && mv "$PRD_FILE.tmp" "$PRD_FILE"

  echo "Marked story $STORY_ID as complete in prd.json"
}

archive_subtasks() {
  if [ ! -f "$SUBTASKS_FILE" ]; then
    return 0
  fi

  STORY_ID=$(jq -r '.storyId' "$SUBTASKS_FILE" 2>/dev/null || echo "unknown")
  DATE=$(date +%Y-%m-%d)
  ARCHIVE_SUBDIR="$ARCHIVE_DIR/$DATE-completed-stories"

  mkdir -p "$ARCHIVE_SUBDIR"
  cp "$SUBTASKS_FILE" "$ARCHIVE_SUBDIR/subtasks-$STORY_ID.json"
  rm "$SUBTASKS_FILE"

  echo "Archived subtasks for $STORY_ID to $ARCHIVE_SUBDIR"
}

trigger_replan() {
  echo "Triggering replan due to blocked subtasks..."
  if [ -f "$SUBTASKS_FILE" ]; then
    # Keep subtasks.json but the planner will see blocked status and replan
    rm "$SUBTASKS_FILE"
  fi
}

# --- Archive Previous Run (if branch changed) ---

if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  LAST_BRANCH=$(cat "$LAST_BRANCH_FILE" 2>/dev/null || echo "")

  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    DATE=$(date +%Y-%m-%d)
    FOLDER_NAME=$(echo "$LAST_BRANCH" | sed 's|^ralph/||')
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"

    echo "Archiving previous run: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$SUBTASKS_FILE" ] && cp "$SUBTASKS_FILE" "$ARCHIVE_FOLDER/"
    echo "   Archived to: $ARCHIVE_FOLDER"

    # Reset for new run
    [ -f "$SUBTASKS_FILE" ] && rm "$SUBTASKS_FILE"
    echo "# Ralph Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
fi

# Track current branch
if [ -f "$PRD_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  if [ -n "$CURRENT_BRANCH" ]; then
    echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
  fi
fi

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Ralph Progress Log" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

echo "Starting Ralph (Planner+Worker) - Max iterations: $MAX_ITERATIONS"
echo ""

# --- Main Loop ---

for i in $(seq 1 $MAX_ITERATIONS); do
  echo "═══════════════════════════════════════════════════════════════"
  echo "  Ralph Iteration $i of $MAX_ITERATIONS"
  echo "═══════════════════════════════════════════════════════════════"
  echo ""

  # PHASE 1: Story Completion Check
  if all_subtasks_complete; then
    echo "[Phase 1] All subtasks complete - finalizing story..."
    mark_story_complete
    archive_subtasks
    echo ""
  fi

  # PHASE 2: Overall Completion Check
  if all_stories_complete; then
    echo ""
    echo "All stories complete!"
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  # PHASE 3: Handle Blocked Subtasks
  if has_blocked_subtasks && ! has_pending_subtasks; then
    echo "[Phase 3] All remaining subtasks are blocked - triggering replan..."
    trigger_replan
  fi

  # PHASE 4: Decide Planner vs Worker
  if has_pending_subtasks; then
    # --- WORKER PHASE ---
    PENDING_COUNT=$(jq '[.subtasks[] | select(.status == "pending")] | length' "$SUBTASKS_FILE")
    STORY_ID=$(jq -r '.storyId' "$SUBTASKS_FILE")
    echo "[Phase 4] Running WORKER agent ($PENDING_COUNT pending subtasks for $STORY_ID)..."
    echo ""

    OUTPUT=$(claude --dangerously-skip-permissions -p "$(cat "$SCRIPT_DIR/worker-prompt.md")" 2>&1 | tee /dev/stderr) || true

    # Reset planner failure count on successful worker run
    PLANNER_FAILURES=0

  else
    # --- PLANNER PHASE ---
    echo "[Phase 4] Running PLANNER agent (no pending subtasks)..."
    echo ""

    OUTPUT=$(claude --dangerously-skip-permissions -p "$(cat "$SCRIPT_DIR/planner-prompt.md")" 2>&1 | tee /dev/stderr) || true

    # Verify planner created subtasks.json
    if [ ! -f "$SUBTASKS_FILE" ]; then
      PLANNER_FAILURES=$((PLANNER_FAILURES + 1))
      echo ""
      echo "WARNING: Planner failed to create subtasks.json (attempt $PLANNER_FAILURES of $MAX_PLANNER_FAILURES)"

      if [ "$PLANNER_FAILURES" -ge "$MAX_PLANNER_FAILURES" ]; then
        echo ""
        echo "FATAL: Planner failed $MAX_PLANNER_FAILURES times consecutively"
        echo "Check the output above for errors."
        exit 1
      fi

      echo "Retrying in next iteration..."
    else
      SUBTASK_COUNT=$(jq '.subtasks | length' "$SUBTASKS_FILE")
      STORY_ID=$(jq -r '.storyId' "$SUBTASKS_FILE")
      echo ""
      echo "Planner created $SUBTASK_COUNT subtasks for story $STORY_ID"
      PLANNER_FAILURES=0
    fi
  fi

  # PHASE 5: Check for early completion signal (edge case)
  if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
    echo ""
    echo "Ralph completed all tasks!"
    exit 0
  fi

  echo ""
  echo "Iteration $i complete. Continuing..."
  echo ""
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all tasks."
echo "Check $PROGRESS_FILE for status."
exit 1

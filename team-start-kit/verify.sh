#!/usr/bin/env bash

set -u

required_files=(
  "README.md"
  "TEAM_REGISTRATION.md"
  "TEAM_CHARTER.md"
  "ROLE_ROTATION.csv"
  "PROJECT_IDEA_CARD.md"
  "INDIVIDUAL_CONTRIBUTION.md"
  "TOPIC_CATALOG.md"
  "COURSE_ROADMAP.md"
  "SKILL_PASSPORT.md"
  "TEAM_FORMATION_BOARD.md"
  "TOPIC_SHORTLIST_SCORECARD.md"
  "RED_TEAM_REVIEW.md"
  "DATA_FEASIBILITY_GATE.md"
  "WEEKLY_SPRINT.md"
  "DECISION_LOG.md"
  "TROUBLESHOOTING_LOG.md"
  "PEER_TEST.md"
  "TEAM_HEALTH_CHECK.md"
)

missing=0
for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "MISSING: $file"
    missing=1
  fi
done

if [[ "$missing" -eq 0 ]]; then
  echo "READY: team-start-kit"
  exit 0
fi

echo "CHECK: ZIP을 다시 풀고 team-start-kit 폴더에서 실행하세요."
exit 1

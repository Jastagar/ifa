#!/usr/bin/env bash
# Sends fake phone notifications to a running Ifa (POST /activate).
# Usage: scripts/notify-test.sh [single|burst|urgent|low|email|all]
# Reads IFA_API_PORT and IFA_API_TOKEN from .env unless they are already set.

set -u
cd "$(dirname "$0")/.."

env_value() { [[ -f .env ]] && grep -E "^[[:space:]]*$1[[:space:]]*=" .env | head -n 1 | sed -E 's/^[^=]*=[[:space:]]*//; s/^["'\'']//; s/["'\'']$//'; }
IFA_API_PORT="${IFA_API_PORT:-$(env_value IFA_API_PORT)}"
IFA_API_TOKEN="${IFA_API_TOKEN:-$(env_value IFA_API_TOKEN)}"
URL="http://127.0.0.1:${IFA_API_PORT:-8787}"

send() {
    curl -s -o /dev/null -w "  %{http_code}  $2 / $3: $4\n" -X POST "$URL/activate" \
        -H "Content-Type: application/json" -H "X-IFA-Token: ${IFA_API_TOKEN}" \
        -d "$(printf '{"context":{"packageName":"%s","appName":"%s","title":"%s","body":"%s"}}' "$1" "$2" "$3" "$4")"
}

single() {
    echo '[single] One WhatsApp message. Expect: listen sound, then ask "What is it?"'
    send com.whatsapp WhatsApp Joe "Are you free tonight?"
}

burst() {
    echo "[burst] Four broken messages from one sender, 1s apart. Expect: grouped into one, announced as urgent."
    send com.whatsapp WhatsApp Mom "Hi"; sleep 1
    send com.whatsapp WhatsApp Mom "How are you"; sleep 1
    send com.whatsapp WhatsApp Mom "Are you free? not anything like a"; sleep 1
    send com.whatsapp WhatsApp Mom "Emergency, call me now"
}

urgent() {
    echo '[urgent] One urgent SMS. Expect: "Sir, there is an urgent message for you."'
    send com.google.android.apps.messaging Messages Dad "Accident on the highway. I am okay but come to the hospital right now"
}

low() {
    echo "[low] Instagram like (low priority). Expect: stored silently, no announcement."
    send com.instagram.android Instagram priya.k "liked your photo"
}

email() {
    echo "[email] Gmail (normal priority). Expect: listen sound unless the LLM rates it urgent."
    send com.google.android.gm Gmail "Rahul Mehta" "Q3 review moved to Monday 10am, please confirm"
}

all() {
    low
    email
    echo "Waiting 6s so the email settles before the next sender..."
    sleep 6
    burst
}

if ! curl -s -m 3 "$URL/health" >/dev/null; then
    echo "Ifa is not listening on $URL. Launch Ifa first."
    exit 1
fi

case "${1:-single}" in
    single|burst|urgent|low|email|all) "${1:-single}" ;;
    *) echo "Unknown scenario '$1'. Use: single, burst, urgent, low, email, all"; exit 1 ;;
esac

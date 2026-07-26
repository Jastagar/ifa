# IfaNotif - Project Context (Resume Tomorrow)

## Goal

Build a bridge between Android notifications and Ifa.

Flow:

```
Android Notification
        ↓
NotificationListenerService
        ↓
Retrofit
        ↓
Spring Boot Bridge
        ↓
Python Activation Server
        ↓
Ifa starts listening
        ↓
User speaks
        ↓
LLM receives notification as conversation context
```

---

# Android

## Stack

- Kotlin
- Jetpack Compose
- NotificationListenerService
- Retrofit
- Coroutines

## Current Status

Notification listener works.

Logs show:

```
Package : org.telegram.messenger
Title   : n8n_user_demo_bot
Message : haan bhai, kya krra hai?
```

Meaning Android successfully extracts notification title and message.

Current service:

```kotlin
NotificationTrackerService
```

Creates:

```kotlin
NotificationRequest(
    packageName,
    title,
    body,
    appName,
    receivedAt,
    deviceId
)
```

Uploads using

```
NotificationRepository.upload()
```

---

# Spring Boot

## Stack

- Java 21
- Spring Boot 3.5.5
- RestClient (currently replaced temporarily)
- PostgreSQL
- Flyway

Notification endpoint receives Android payload.

Current NotificationService creates:

```java
ActivationRequest(
    """
    A new notification has arrived.

    The notification is provided in the activation context.

    Use it as reference for the conversation.
    """,
    request,
    false
)
```

Then calls

```
IfaService.activate()
```

---

# Python

Current API:

```
POST /activate
```

Goal is NOT to immediately ask LLM.

Instead it should start microphone listening.

Current code:

```python
payload = json.loads(...)
input_mode._listener.start_listening_from_api(
    api_context=json.dumps(
        payload.get("context"),
        ensure_ascii=False
    )
)
```

Returns

```
202 Accepted
```

---

# Major issue solved today

RestClient was sending

```
Transfer-Encoding: chunked
```

Python server expected

```
Content-Length
```

So Python printed

```
request body is missing
```

Headers showed:

```
Transfer-Encoding: chunked
Content-Length = 0
```

We temporarily bypassed RestClient.

Current IfaService uses

```
HttpURLConnection
```

with

```java
setFixedLengthStreamingMode(...)
```

instead of RestClient.

Long term we should migrate Python API to Flask/FastAPI.

---

# Current unresolved issue

Android logs

```
Message : haan bhai, kya krra hai?
```

But inside Python context the body becomes

```
null
```

This means somewhere between

Android
→ Retrofit
→ Spring Boot
→ ActivationRequest
→ Python

the message/body field is being lost.

Need to inspect DTO mapping.

Most likely cause:

Different field names like

```
body
```

vs

```
text
```

Need to verify

- Android NotificationRequest.kt
- Spring NotificationRequest.java
- ActivationRequest serialization

---

# Android deployment

Successfully built APK.

Installed on physical phone.

Need to:

Settings

↓

Notification Access

↓

Enable Ifanotif

NOT

```
POST_NOTIFICATIONS
```

Notification permission is unrelated.

NotificationListenerService uses

```
Notification Access
```

which is a Special App Access.

---

# Future improvements

Instead of

```
BaseHTTPRequestHandler
```

replace activation server with

```
FastAPI
```

Benefits

- automatic JSON
- chunked encoding
- validation
- future endpoints
- less maintenance

Not priority now.

---

# Next session TODO

## High Priority

- Find why notification body becomes null inside Python.
- Compare Android DTO and Spring DTO.
- Print JSON received by Spring.
- Print JSON sent from Spring to Python.
- Verify ActivationRequest serialization.

## Medium Priority

- Replace HttpURLConnection with proper HTTP client after Python API supports chunked requests.
- Move Python API to FastAPI.

## Low Priority

- Polish Android UI.
- Add backend health check.
- Add better logs.
- Add reconnect handling.

---

# Expected final flow

Telegram notification

↓

Spring Boot receives notification

↓

Python activation server receives notification context

↓

Ifa begins listening automatically

↓

User speaks naturally

↓

LLM answers with notification already in context

No wake word required.
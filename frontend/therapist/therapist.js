// frontend/therapist/therapist.js

document.addEventListener("DOMContentLoaded", () => {
  requireRole(["therapist"]);

  if (document.getElementById("schedule")) {
    initTherapistDashboard();
  }

  if (document.getElementById("reports-table")) {
    initTherapistReports();
  }

  if (document.getElementById("notifications")) {
    initTherapistNotifications();
  }

  if (document.getElementById("therapist-feedback-form")) {
    initTherapistFeedback();
  }
});

async function initTherapistDashboard() {
  const tbody = document.getElementById("schedule");
  tbody.innerHTML = "";

  try {
    const therapistId = getCurrentUserId();
    const sessions = await apiGet(`/api/therapist/schedule?therapist_id=${encodeURIComponent(therapistId)}`);

    sessions.forEach((s) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${s.user_id}</td>
        <td>${s.scheduled_for}</td>
        <td>${s.status}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
  }
}

async function initTherapistReports() {
  const tbody = document.getElementById("reports-table");
  tbody.innerHTML = "";

  try {
    const therapistId = getCurrentUserId();
    const reports = await apiGet(`/api/therapist/reports?therapist_id=${encodeURIComponent(therapistId)}`);

    reports.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.case_id}</td>
        <td>${r.user_id}</td>
        <td>${r.created_at}</td>
        <td><a href="${r.report_url}" target="_blank">عرض</a></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
  }
}

async function initTherapistNotifications() {
  const container = document.getElementById("notifications");
  container.innerHTML = "";

  try {
    const therapistId = getCurrentUserId();
    const notifications = await apiGet(`/api/therapist/notifications?therapist_id=${encodeURIComponent(therapistId)}`);

    if (!notifications.length) {
      container.innerHTML = "<p>لا توجد إشعارات جديدة.</p>";
      return;
    }

    notifications.forEach((n) => {
      const div = document.createElement("div");
      div.className = "support-box";
      div.innerHTML = `
        <h3>${n.title}</h3>
        <p>${n.body}</p>
        <p><small>${n.created_at}</small></p>
      `;
      container.appendChild(div);
    });
  } catch (err) {
    console.error(err);
  }
}

function initTherapistFeedback() {
  const form = document.getElementById("therapist-feedback-form");
  const riskSelect = document.getElementById("risk-level");
  const notesEl = document.getElementById("therapist-notes");
  const sessionIdEl = document.getElementById("session-id");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
      therapist_id: getCurrentUserId(),
      session_id: sessionIdEl.value || null,
      risk_level: riskSelect.value,
      notes: (notesEl.value || "").trim() || null,
    };

    try {
      await apiPost("/api/therapist/feedback", payload);
      alert("تم حفظ الملاحظات.");
      notesEl.value = "";
    } catch (err) {
      console.error(err);
      alert("تعذر حفظ الملاحظات.");
    }
  });
}

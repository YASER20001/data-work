// frontend/admin/admin.js

document.addEventListener("DOMContentLoaded", () => {
  // Only admins are allowed here
  requireRole(["admin"]);

  // Detect which admin page we’re on by checking for specific elements
  const statUsersEl = document.getElementById("stat-users");
  const usersTableBody = document.getElementById("user-table");

  if (statUsersEl) {
    initAdminDashboard();
  }

  if (usersTableBody) {
    initAdminUsers();
  }
});

// ------ Dashboard logic ------
async function initAdminDashboard() {
  try {
    const stats = await apiGet("/api/admin/stats");
    document.getElementById("stat-users").innerText = stats.total_users ?? "--";
    document.getElementById("stat-sessions").innerText = stats.total_sessions ?? "--";
    document.getElementById("stat-reports").innerText = stats.total_reports ?? "--";
  } catch (err) {
    console.error(err);
  }
}

// ------ Users management ------
async function initAdminUsers() {
  const emailInput = document.querySelector('input[type="email"]');
  const passwordInput = document.querySelector('input[type="password"]');
  const roleSelect = document.querySelector("select");
  const createBtn = document.querySelector(".cta-button");
  const tableBody = document.getElementById("user-table");

  createBtn.addEventListener("click", async () => {
    const email = (emailInput.value || "").trim();
    const password = (passwordInput.value || "").trim();
    const role = roleSelect.value;

    if (!email || !password) {
      alert("يرجى إدخال البريد الإلكتروني وكلمة المرور");
      return;
    }

    try {
      const created = await apiPost("/api/admin/users", {
        email,
        password,
        role,
      });
      // Re-load table
      await loadUsers(tableBody);
      emailInput.value = "";
      passwordInput.value = "";
    } catch (err) {
      console.error(err);
      alert("تعذر إنشاء الحساب");
    }
  });

  await loadUsers(tableBody);
}

async function loadUsers(tableBody) {
  tableBody.innerHTML = "";
  try {
    const users = await apiGet("/api/admin/users");
    users.forEach((u) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${u.email}</td>
        <td>${u.role}</td>
        <td><!-- TODO: add deactivate/delete buttons --></td>
      `;
      tableBody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
  }
}

// frontend/roles.js

// Base URL for your API (same idea as chat.html, but without the broken `||` chain)
const API_BASE =
  localStorage.getItem("RIFD_API") ||
  "http://127.0.0.1:8000"; // dev default; override RIFD_API in prod

// ---- Role helpers ----
function getCurrentRole() {
  return localStorage.getItem("rifd_role"); // 'user' | 'therapist' | 'admin'
}

function getCurrentUserId() {
  return localStorage.getItem("rifd_user_id") || "anon";
}

function isLoggedIn() {
  return localStorage.getItem("isLoggedIn") === "true";
}

/**
 * Guard a page by role.
 *
 * allowedRoles: array of roles, e.g. ['admin'], ['therapist'], ['user']
 */
function requireRole(allowedRoles) {
  const role = getCurrentRole();
  if (!isLoggedIn() || !role || (allowedRoles && !allowedRoles.includes(role))) {
    // From /frontend/admin/... etc this points back up to login
    window.location.href = "../login.html";
  }
}

// Simple fetch helper with JSON
async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} failed: ${res.status} - ${text.slice(0, 200)}`);
  }
  return res.json();
}

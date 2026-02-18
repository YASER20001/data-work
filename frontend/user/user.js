// frontend/user/user.js

document.addEventListener("DOMContentLoaded", () => {
  // Only users
  requireRole(["user"]);

  if (document.getElementById("rating-buttons")) {
    initUserFeedback();
  }

  // Emergency page is mostly static; if you add JS later, hook here
});

function initUserFeedback() {
  let selectedRating = null;
  const ratingButtons = document.querySelectorAll(".rating-button");
  const commentEl = document.getElementById("feedback-comment");
  const submitBtn = document.getElementById("submit-feedback");

  ratingButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedRating = parseInt(btn.dataset.rating, 10);
      ratingButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });

  submitBtn.addEventListener("click", async () => {
    if (!selectedRating) {
      alert("يرجى اختيار تقييم");
      return;
    }
    const comment = (commentEl.value || "").trim();
    const sessionId = localStorage.getItem("rifd_session_id") || null;
    const userId = getCurrentUserId();

    try {
      await apiPost("/api/user/feedback", {
        user_id: userId,
        session_id: sessionId,
        rating: selectedRating,
        comment: comment || null,
      });
      alert("شكرًا لتقييمك.");
      commentEl.value = "";
      selectedRating = null;
      ratingButtons.forEach((b) => b.classList.remove("active"));
    } catch (err) {
      console.error(err);
      alert("تعذر إرسال التقييم الآن.");
    }
  });
}

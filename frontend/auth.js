// هذا الكود سينفذ نفسه بمجرد تحميل أي صفحة
document.addEventListener('DOMContentLoaded', function() {

    // التحقق هل المستخدم مسجل دخوله من خلال الذاكرة المحلية للمتصفح
    const isLoggedIn = localStorage.getItem('isLoggedIn');

    if (isLoggedIn === 'true') {
        // إذا كان مسجل دخوله، قم بتغيير القائمة
        updateNavForLoggedInUser();
    }
});

function updateNavForLoggedInUser() {
    const navLinks = document.getElementById('nav-links');
    const loginLink = document.getElementById('login-link');

    // 1. إزالة زر "تسجيل الدخول"
    if (loginLink) {
        loginLink.remove();
    }

    // 2. إنشاء وإضافة رابط "مواعيدي"
    const appointmentsLi = document.createElement('li');
    appointmentsLi.innerHTML = `<a href="appointments.html">مواعيدي</a>`;
    navLinks.appendChild(appointmentsLi);

    // 3. إنشاء وإضافة قائمة "حسابي" المنسدلة
    const accountLi = document.createElement('li');
    accountLi.classList.add('dropdown'); // لإضافة تنسيقات خاصة
    accountLi.innerHTML = `
        <a href="#">حسابي</a>
        <div class="dropdown-content">
            <a href="#">معلوماتي الشخصية</a>
            <a href="#" id="logout-button">تسجيل الخروج</a>
        </div>
    `;
    navLinks.appendChild(accountLi);

    // 4. تفعيل زر تسجيل الخروج
    const logoutButton = document.getElementById('logout-button');
    logoutButton.addEventListener('click', function(event) {
        event.preventDefault();
        
        localStorage.removeItem('isLoggedIn');
        localStorage.removeItem('rifd_user_id');
        localStorage.removeItem('rifd_role');
        localStorage.removeItem('rifd_session_id');
        
        alert('تم تسجيل الخروج بنجاح.');
        window.location.href = 'index.html';
    });
}
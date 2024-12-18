document.addEventListener('DOMContentLoaded', function () {
    const viewReportsButton = document.getElementById('viewReportsButton');
    const warningBar = document.getElementById('warningMessage');

    viewReportsButton.addEventListener('click', function (event) {
        const userRole = this.getAttribute('data-role'); // Check user's role
        if (userRole !== 'admin') {
            event.preventDefault(); // Prevent navigation
            warningBar.style.display = 'block'; // Show warning message

            // Hide warning message after 3 seconds
            setTimeout(() => {
                warningBar.style.display = 'none';
            }, 3000);
        } else {
            // Redirect to admin page if role is admin
            window.location.href = '/admin';
        }
    });
});

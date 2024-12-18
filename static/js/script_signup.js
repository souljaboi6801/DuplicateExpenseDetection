document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");

    form.addEventListener("submit", function (e) {
        const password = document.querySelector("#password").value;
        const confirmPassword = document.querySelector("#confirm_password").value;

        if (password !== confirmPassword) {
            alert("Passwords do not match!");
            e.preventDefault(); // Prevent form submission
        }
    });
});

document.addEventListener("DOMContentLoaded", function () {
    const modal = document.getElementById("imageModal");
    const modalImage = document.getElementById("modalImage");
    const closeModal = document.querySelector(".close");

    // Open modal when viewing images
    document.querySelectorAll(".view-image").forEach(button => {
        button.addEventListener("click", function (event) {
            event.preventDefault();
            modalImage.src = this.getAttribute("data-image");
            modal.style.display = "block";
        });
    });

    // Close modal when 'X' is clicked
    closeModal.addEventListener("click", function () {
        modal.style.display = "none";
        modalImage.src = "";
    });

    // Close modal when clicking outside the image
    modal.addEventListener("click", function (event) {
        if (event.target === modal) {
            modal.style.display = "none";
            modalImage.src = "";
        }
    });
});

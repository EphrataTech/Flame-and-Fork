document.addEventListener("DOMContentLoaded", () => {
  // Initialize all functionality
  initDarkMode()
  initMobileMenu()
  initCarousel()
  initTypewriter()
  initSmoothScrolling()
  initFormValidation()
})

// Dark Mode Functionality
function initDarkMode() {
  const darkModeToggle = document.getElementById("darkModeToggle")
  const html = document.documentElement

  const savedTheme = localStorage.getItem("theme")
  if (savedTheme) {
    html.classList.toggle("dark", savedTheme === "dark")
  }

  if (darkModeToggle) {
    darkModeToggle.addEventListener("click", () => {
      html.classList.toggle("dark")
      const isDark = html.classList.contains("dark")
      localStorage.setItem("theme", isDark ? "dark" : "light")
    })
  }
}
// Mobile Menu Functionality
function initMobileMenu() {
  const mobileMenuToggle = document.getElementById("mobileMenuToggle")
  const mobileMenu = document.getElementById("mobileMenu")
  const mobileMenuDropdown = document.getElementById("mobileMenuDropdown")
  const mobileDropdownContent = document.getElementById("mobileDropdownContent")

  if (mobileMenuToggle && mobileMenu) {
    mobileMenuToggle.addEventListener("click", () => {
      mobileMenu.classList.toggle("hidden")
      mobileMenu.classList.toggle("show")
    })
  }

  if (mobileMenuDropdown && mobileDropdownContent) {
    mobileMenuDropdown.addEventListener("click", () => {
      mobileDropdownContent.classList.toggle("hidden")
      const icon = mobileMenuDropdown.querySelector("i")
      icon.classList.toggle("fa-chevron-down")
      icon.classList.toggle("fa-chevron-up")
    })
  }
}
// Carousel Functionality
function initCarousel() {
  const slides = document.querySelectorAll(".carousel-slide")
  const dots = document.querySelectorAll(".carousel-dot")
  let currentSlide = 0

  if (slides.length === 0) return

  function showSlide(index) {
    // Remove active class from all slides and dots
    slides.forEach((slide) => slide.classList.remove("active"))
    dots.forEach((dot) => dot.classList.remove("active"))

    // Add active class to current slide and dot
    if (slides[index]) slides[index].classList.add("active")
    if (dots[index]) dots[index].classList.add("active")
  }

  function nextSlide() {
    currentSlide = (currentSlide + 1) % slides.length
    showSlide(currentSlide)
  }

  // Auto-play carousel
  setInterval(nextSlide, 5000)

  // Dot navigation
  dots.forEach((dot, index) => {
    dot.addEventListener("click", () => {
      currentSlide = index
      showSlide(currentSlide)
    })
  })
}


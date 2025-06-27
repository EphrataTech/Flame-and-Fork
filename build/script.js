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


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

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

// Smooth Scrolling
function initSmoothScrolling() {
  const links = document.querySelectorAll('a[href^="#"]')

  links.forEach((link) => {
    link.addEventListener("click", function (e) {
      const href = this.getAttribute("href")
      if (href === "#") return

      const target = document.querySelector(href)
      if (target) {
        e.preventDefault()
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        })
      }
    })
  })
}

// Form Validation
function initFormValidation() {
  const feedbackForm = document.getElementById("feedbackForm")
  if (!feedbackForm) return

  const fields = {
    fullName: {
      element: document.getElementById("fullName"),
      error: document.getElementById("fullNameError"),
      validate: (value) => {
        if (!value.trim()) return "Full name is required"
        if (value.trim().length < 2) return "Full name must be at least 2 characters"
        return null
      },
    },
    email: {
      element: document.getElementById("email"),
      error: document.getElementById("emailError"),
      validate: (value) => {
        if (!value.trim()) return "Email is required"
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
        if (!emailRegex.test(value)) return "Please enter a valid email address"
        return null
      },
    },
    phone: {
      element: document.getElementById("phone"),
      error: document.getElementById("phoneError"),
      validate: (value) => {
        if (value.trim() && !/^[+]?[0-9\s\-()]{10,}$/.test(value)) {
          return "Please enter a valid phone number"
        }
        return null
      },
    },
    message: {
      element: document.getElementById("message"),
      error: document.getElementById("messageError"),
      validate: (value) => {
        if (!value.trim()) return "Message is required"
        if (value.trim().length < 10) return "Message must be at least 10 characters"
        return null
      },
    },
  }

}
// Real-time validation
  Object.keys(fields).forEach((fieldName) => {
    const field = fields[fieldName]
    if (field.element) {
      field.element.addEventListener("blur", () => {
        validateField(fieldName, field)
      })

      field.element.addEventListener("input", () => {
        if (field.error && !field.error.classList.contains("hidden")) {
          validateField(fieldName, field)
        }
      })
    }
  })

  function validateField(fieldName, field) {
    const error = field.validate(field.element.value)
    if (error) {
      field.error.textContent = error
      field.error.classList.remove("hidden")
      field.element.classList.add("border-red-500")
      return false
    } else {
      field.error.classList.add("hidden")
      field.element.classList.remove("border-red-500")
      return true
    }
  }
  // Form submission
  feedbackForm.addEventListener("submit", (e) => {
    e.preventDefault()

    let isValid = true
    Object.keys(fields).forEach((fieldName) => {
      const field = fields[fieldName]
      if (!validateField(fieldName, field)) {
        isValid = false
      }
    })

    if (isValid) {
      showSuccessModal()
      feedbackForm.reset()
    }
  })


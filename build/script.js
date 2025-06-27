// Restaurant Website JavaScript - Flame & Fork
// All interactive functionality including dark mode, carousel, typewriter, forms, and mobile navigation

document.addEventListener("DOMContentLoaded", function () {
  // ==================== VARIABLES ====================
  let isDarkMode = false;
  let currentSlide = 0;
  let typewriterIndex = 0;
  let charIndex = 0;
  let isDeleting = false;

  const heroImages = [
    "https://images.pexels.com/photos/958545/pexels-photo-958545.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&fit=crop",
    "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&fit=crop",
    "https://images.pexels.com/photos/941861/pexels-photo-941861.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&fit=crop",
  ];

  const typewriterTexts = ["RESTAURANT", "EXPERIENCE", "FLAVORS"];

  // ==================== DOM ELEMENTS ====================
  const darkModeToggle = document.getElementById("darkModeToggle");
  const mobileMenuToggle = document.getElementById("mobileMenuToggle");
  const mobileMenu = document.getElementById("mobileMenu");
  const mobileMenuDropdown = document.getElementById("mobileMenuDropdown");
  const mobileDropdownContent = document.getElementById(
    "mobileDropdownContent"
  );
  const heroSection = document.querySelector(
    'section[style*="background-image"]'
  );
  const carouselDots = document.querySelectorAll(".carousel-dot");
  const typewriterElement = document.getElementById("typewriter");
  const feedbackForm = document.getElementById("feedbackForm");
  const successModal = document.getElementById("successModal");
  const closeModalBtn = document.getElementById("closeModal");

  // ==================== DARK MODE FUNCTIONALITY ====================
  function toggleDarkMode() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle("dark", isDarkMode);

    // Update dark mode toggle icon
    const moonIcon = darkModeToggle.querySelector(".fa-moon");
    const sunIcon = darkModeToggle.querySelector(".fa-sun");

    if (isDarkMode) {
      moonIcon.classList.add("hidden");
      sunIcon.classList.remove("hidden");
    } else {
      moonIcon.classList.remove("hidden");
      sunIcon.classList.add("hidden");
    }

    // Save preference to localStorage
    localStorage.setItem("darkMode", isDarkMode);
  }

  // Load dark mode preference
  function loadDarkModePreference() {
    const savedDarkMode = localStorage.getItem("darkMode");
    if (savedDarkMode === "true") {
      isDarkMode = true;
      toggleDarkMode();
    }
  }

  // ==================== MOBILE NAVIGATION ====================
  function toggleMobileMenu() {
    const isOpen = mobileMenu.classList.contains("max-h-0");

    if (isOpen) {
      mobileMenu.classList.remove("hidden", "max-h-0");
      mobileMenu.classList.add("max-h-[500px]");
      mobileMenuToggle.innerHTML = '<i class="fas fa-times"></i>';
    } else {
      mobileMenu.classList.add("max-h-0");
      mobileMenu.classList.remove("max-h-[500px]");
      setTimeout(() => {
        mobileMenu.classList.add("hidden");
      }, 300); // Match duration-300 from Tailwind
      mobileMenuToggle.innerHTML = '<i class="fas fa-bars"></i>';
    }
  }

  function toggleMobileDropdown() {
    const isOpen = !mobileDropdownContent.classList.contains("hidden");

    if (isOpen) {
      mobileDropdownContent.classList.add("hidden");
    } else {
      mobileDropdownContent.classList.remove("hidden");
    }
  }

  // ==================== HERO CAROUSEL ====================
  function updateCarousel() {
    if (heroSection) {
      const imageUrl = heroImages[currentSlide];
      heroSection.style.backgroundImage = `linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url(${imageUrl})`;

      // Update carousel dots
      carouselDots.forEach((dot, index) => {
        if (index === currentSlide) {
          dot.classList.add("active");
          dot.style.backgroundColor = "white";
        } else {
          dot.classList.remove("active");
          dot.style.backgroundColor = "rgba(255, 255, 255, 0.5)";
        }
      });
    }
  }

  function nextSlide() {
    currentSlide = (currentSlide + 1) % heroImages.length;
    updateCarousel();
  }

  function goToSlide(slideIndex) {
    currentSlide = slideIndex;
    updateCarousel();
  }

  // Auto-play carousel
  function startCarousel() {
    setInterval(nextSlide, 5000); // Change slide every 5 seconds
  }

  // ==================== TYPEWRITER EFFECT ====================
  function typeWriter() {
    const currentText = typewriterTexts[typewriterIndex];

    if (isDeleting) {
      typewriterElement.textContent = currentText.substring(0, charIndex - 1);
      charIndex--;
    } else {
      typewriterElement.textContent = currentText.substring(0, charIndex + 1);
      charIndex++;
    }

    let typeSpeed = 150;
    if (isDeleting) typeSpeed /= 2;

    if (!isDeleting && charIndex === currentText.length) {
      typeSpeed = 2000; // Pause at end
      isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
      isDeleting = false;
      typewriterIndex = (typewriterIndex + 1) % typewriterTexts.length;
      typeSpeed = 500; // Pause before next word
    }

    setTimeout(typeWriter, typeSpeed);
  }

  // ==================== FORM VALIDATION & SUBMISSION ====================
  function validateForm(formData) {
    const errors = {};

    // Name validation
    if (!formData.fullName.trim()) {
      errors.fullName = "Name is required";
    }

    // Email validation
    if (!formData.email.trim()) {
      errors.email = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      errors.email = "Please enter a valid email address";
    }

    // Message validation
    if (!formData.message.trim()) {
      errors.message = "Message is required";
    }

    return errors;
  }

  function displayFormErrors(errors) {
    // Clear previous errors
    document.querySelectorAll(".error-message").forEach((error) => {
      error.classList.add("hidden");
      error.textContent = "";
    });

    // Display new errors
    Object.keys(errors).forEach((field) => {
      const errorElement = document.getElementById(field + "Error");
      if (errorElement) {
        errorElement.textContent = errors[field];
        errorElement.classList.remove("hidden");
      }
    });
  }

  function handleFormSubmission(e) {
    e.preventDefault();

    // Get form data
    const formData = {
      fullName: document.getElementById("fullName").value,
      email: document.getElementById("email").value,
      phone: document.getElementById("phone").value,
      message: document.getElementById("message").value,
    };

    // Validate form
    const errors = validateForm(formData);

    if (Object.keys(errors).length > 0) {
      displayFormErrors(errors);
      return;
    }

    // Clear any existing errors
    displayFormErrors({});

    // Simulate form submission (replace with actual API call)
    console.log("Form submitted:", formData);

    // Show success modal
    showSuccessModal();

    // Reset form
    feedbackForm.reset();
  }

  function showSuccessModal() {
    successModal.classList.remove("hidden");
    successModal.classList.add("flex");
    document.body.style.overflow = "hidden"; // Prevent background scrolling
  }

  function hideSuccessModal() {
    successModal.classList.add("hidden");
    successModal.classList.remove("flex");
    document.body.style.overflow = "auto"; // Restore scrolling
  }

  // ==================== SMOOTH SCROLLING ====================
  function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener("click", function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute("href"));
        if (target) {
          target.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });

          // Close mobile menu if open
          if (!mobileMenu.classList.contains("hidden")) {
            toggleMobileMenu();
          }
        }
      });
    });
  }

  // ==================== DISH CARD ANIMATIONS ====================
  function initDishCardAnimations() {
    const dishCards = document.querySelectorAll(".dish-card");

    const observerOptions = {
      threshold: 0.1,
      rootMargin: "0px 0px -50px 0px",
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = "1";
          entry.target.style.transform = "translateY(0)";
        }
      });
    }, observerOptions);

    dishCards.forEach((card) => {
      card.style.opacity = "0";
      card.style.transform = "translateY(20px)";
      card.style.transition = "opacity 0.6s ease, transform 0.6s ease";
      observer.observe(card);
    });
  }

  // ==================== NAVBAR SCROLL EFFECT ====================
  function initNavbarScrollEffect() {
    const header = document.querySelector("header");
    let lastScrollTop = 0;

    window.addEventListener("scroll", () => {
      const scrollTop =
        window.pageYOffset || document.documentElement.scrollTop;

      if (scrollTop > 100) {
        header.style.backgroundColor = "rgba(0, 0, 0, 0.95)";
      } else {
        header.style.backgroundColor = "rgba(0, 0, 0, 0.7)";
      }

      lastScrollTop = scrollTop;
    });
  }

  // ==================== EVENT LISTENERS ====================
  function initEventListeners() {
    // Dark mode toggle
    if (darkModeToggle) {
      darkModeToggle.addEventListener("click", toggleDarkMode);
    }

    // Mobile menu toggle
    if (mobileMenuToggle) {
      mobileMenuToggle.addEventListener("click", toggleMobileMenu);
    }

    // Mobile dropdown toggle
    if (mobileMenuDropdown) {
      mobileMenuDropdown.addEventListener("click", toggleMobileDropdown);
    }

    // Carousel dots
    carouselDots.forEach((dot, index) => {
      dot.addEventListener("click", () => goToSlide(index));
    });

    // Form submission
    if (feedbackForm) {
      feedbackForm.addEventListener("submit", handleFormSubmission);
    }

    // Modal close
    if (closeModalBtn) {
      closeModalBtn.addEventListener("click", hideSuccessModal);
    }

    // Close modal when clicking outside
    if (successModal) {
      successModal.addEventListener("click", (e) => {
        if (e.target === successModal) {
          hideSuccessModal();
        }
      });
    }

    // Close modal with Escape key
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !successModal.classList.contains("hidden")) {
        hideSuccessModal();
      }
    });

    // Close mobile menu when clicking outside
    document.addEventListener("click", (e) => {
      if (
        !e.target.closest("nav") &&
        !mobileMenu.classList.contains("hidden")
      ) {
        toggleMobileMenu();
      }
    });
  }

  // ==================== INITIALIZATION ====================
  function init() {
    // Load user preferences
    loadDarkModePreference();

    // Initialize carousel
    updateCarousel();
    startCarousel();

    // Start typewriter effect
    if (typewriterElement) {
      typeWriter();
    }

    // Initialize animations and effects
    initSmoothScrolling();
    initDishCardAnimations();
    initNavbarScrollEffect();

    // Set up event listeners
    initEventListeners();

    console.log("Flame & Fork website initialized successfully! 🔥🍴");
  }

  // Start the application
  init();
});

// ==================== UTILITY FUNCTIONS ====================

// Debounce function for performance optimization
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Format phone number (optional enhancement)
function formatPhoneNumber(phoneNumber) {
  const cleaned = phoneNumber.replace(/\D/g, "");
  const match = cleaned.match(/^(\d{3})(\d{3})(\d{4})$/);
  if (match) {
    return `(${match[1]}) ${match[2]}-${match[3]}`;
  }
  return phoneNumber;
}

// Add loading states for better UX
function showLoading(element) {
  element.disabled = true;
  element.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Loading...';
}

function hideLoading(element, originalText) {
  element.disabled = false;
  element.innerHTML = originalText;
}

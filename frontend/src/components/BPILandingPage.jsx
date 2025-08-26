import { 
  CreditCard, 
  Shield, 
  Smartphone, 
  TrendingUp, 
  Users, 
  Globe,
  ArrowRight,
  Play,
  Star,
  CheckCircle,
  Building,
  Banknote
} from 'lucide-react'
import { useState, useEffect } from 'react'

const services = [
  {
    icon: <CreditCard className="w-8 h-8" />,
    title: "Credit Cards",
    description: "Discover our range of credit cards with exclusive rewards and benefits",
    image: "/api/placeholder/600/300"
  },
  {
    icon: <Building className="w-8 h-8" />,
    title: "Loans & Mortgages", 
    description: "Flexible loan options to help you achieve your dreams",
    image: "/api/placeholder/600/300"
  },
  {
    icon: <TrendingUp className="w-8 h-8" />,
    title: "Investments",
    description: "Grow your wealth with our investment and insurance products",
    image: "/api/placeholder/600/300"
  },
  {
    icon: <Smartphone className="w-8 h-8" />,
    title: "Digital Banking",
    description: "Bank anytime, anywhere with our mobile and online platforms",
    image: "/api/placeholder/600/300"
  }
]

const BPILandingPage = () => {
  const [isLoaded, setIsLoaded] = useState(false)
  const [activeService, setActiveService] = useState(0)

  useEffect(() => {
    setIsLoaded(true)
    
    // Auto-rotate services
    const interval = setInterval(() => {
      setActiveService(prev => (prev + 1) % services.length)
    }, 4000)
    
    return () => clearInterval(interval)
  }, [])
  const features = [
    {
      icon: <Shield className="w-6 h-6" />,
      title: "Bank-Grade Security",
      description: "Your money and data are protected with world-class security"
    },
    {
      icon: <Users className="w-6 h-6" />,
      title: "24/7 Support",
      description: "Get help anytime with our round-the-clock customer service"
    },
    {
      icon: <Globe className="w-6 h-6" />,
      title: "Global Network",
      description: "Access your account worldwide with our extensive ATM network"
    }
  ]

  return (
    <div className="bpi-landing">
      {/* Navigation */}
      <nav className="bpi-nav">
        <div className="nav-container">
          <div className="nav-logo">
            <img src="/header.png" alt="BPI Logo" className="logo-img" />
          </div>
          
          <div className="nav-menu">
            <a href="#personal">Personal Banking</a>
            <a href="#business">Business Banking</a>
            <a href="#investments">Investments</a>
            <a href="#loans">Loans</a>
            <a href="#cards">Cards</a>
          </div>

          <div className="nav-actions">
            <button className="nav-login">Log In</button>
            <button className="nav-register">Open Account</button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-container">
          <div className="hero-content">
            <div className={`hero-text ${isLoaded ? 'loaded' : ''}`}>
              <h1 className="hero-title">
                Banking Made 
                <span className="gradient-text"> Simple</span>
              </h1>
              <p className="hero-subtitle">
                Experience the future of banking with BPI's innovative digital solutions. 
                Manage your finances, invest wisely, and grow your wealth with the Philippines' 
                most trusted bank.
              </p>
              
              <div className="hero-actions">
                <button className="cta-primary">
                  <Banknote className="w-5 h-5" />
                  Open Account
                  <ArrowRight className="w-5 h-5" />
                </button>
                <button className="cta-secondary">
                  <Play className="w-5 h-5" />
                  Watch Demo
                </button>
              </div>

              <div className="hero-stats">
                <div className="stat">
                  <div className="stat-number">12M+</div>
                  <div className="stat-label">Customers</div>
                </div>
                <div className="stat">
                  <div className="stat-number">1,800+</div>
                  <div className="stat-label">Branches</div>
                </div>
                <div className="stat">
                  <div className="stat-number">170+</div>
                  <div className="stat-label">Years</div>
                </div>
              </div>
            </div>
          </div>

          <div className="hero-visual">
            <div className="hero-card">
              <div className="card-glow"></div>
              <img src="/LandingBanner.svg" alt="BPI Credit Card" className="card-image" />
            </div>
          </div>
        </div>
      </section>

      {/* Services Section */}
      <section className="services-section">
        <div className="services-container">
          <div className="section-header">
            <h2>Our Banking Services</h2>
            <p>Comprehensive financial solutions tailored to your needs</p>
          </div>

          <div className="services-showcase">
            <div className="services-nav">
              {services.map((service, index) => (
                <button
                  key={index}
                  className={`service-tab ${activeService === index ? 'active' : ''}`}
                  onClick={() => setActiveService(index)}
                >
                  {service.icon}
                  <div>
                    <h3>{service.title}</h3>
                    <p>{service.description}</p>
                  </div>
                </button>
              ))}
            </div>

            <div className="services-display">
              <div className="service-visual">
                <img 
                  src={services[activeService].image} 
                  alt={services[activeService].title}
                  className="service-image"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="features-container">
          <div className="section-header">
            <h2>Why Choose BPI?</h2>
            <p>Experience banking excellence with the Philippines' most trusted financial institution</p>
          </div>

          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon">
                  {feature.icon}
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-container">
          <div className="cta-content">
            <h2>Ready to Start Banking with BPI?</h2>
            <p>Join millions of Filipinos who trust BPI for their banking needs</p>
            
            <div className="cta-actions">
              <button className="cta-primary">
                Open Your Account Today
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>

            <div className="trust-indicators">
              <div className="indicator">
                <CheckCircle className="w-5 h-5" />
                <span>PDIC Insured</span>
              </div>
              <div className="indicator">
                <Star className="w-5 h-5" />
                <span>5-Star Rated</span>
              </div>
              <div className="indicator">
                <Shield className="w-5 h-5" />
                <span>Bank-Grade Security</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bpi-footer">
        <div className="footer-container">
          <div className="footer-grid">
            <div className="footer-section">
              <img src="/api/placeholder/120/40" alt="BPI Logo" className="footer-logo" />
              <p>The Bank of the Philippine Islands - your partner in building a better future.</p>
            </div>
            
            <div className="footer-section">
              <h4>Personal Banking</h4>
              <ul>
                <li><a href="#">Savings Account</a></li>
                <li><a href="#">Credit Cards</a></li>
                <li><a href="#">Personal Loans</a></li>
                <li><a href="#">Home Loans</a></li>
              </ul>
            </div>

            <div className="footer-section">
              <h4>Digital Services</h4>
              <ul>
                <li><a href="#">BPI Online</a></li>
                <li><a href="#">Mobile App</a></li>
                <li><a href="#">Digital Wallet</a></li>
                <li><a href="#">Online Investment</a></li>
              </ul>
            </div>

            <div className="footer-section">
              <h4>Support</h4>
              <ul>
                <li><a href="#">Contact Us</a></li>
                <li><a href="#">Find Branch</a></li>
                <li><a href="#">FAQs</a></li>
                <li><a href="#">Security Tips</a></li>
              </ul>
            </div>
          </div>

          <div className="footer-bottom">
            <p>&copy; 2025 Bank of the Philippine Islands. All rights reserved.</p>
            <div className="footer-links">
              <a href="#">Privacy Policy</a>
              <a href="#">Terms of Service</a>
              <a href="#">Sitemap</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default BPILandingPage
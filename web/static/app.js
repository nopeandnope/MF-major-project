// DOM Elements with error handling
const form = document.getElementById('investmentForm');
const optionsSection = document.getElementById('optionsSection');
const allOptionsContainer = document.getElementById('allOptionsContainer');
const selectedOptionSection = document.getElementById('selectedOptionSection');
const selectedOptionDetails = document.getElementById('selectedOptionDetails');
const detailed = document.getElementById('detailedRecommendations');
const recommendationsSection = document.getElementById('recommendationsSection');

// Validate that critical elements exist
function validateRequiredElements() {
  const requiredElements = {
    form: 'investmentForm',
    optionsSection: 'optionsSection',
    allOptionsContainer: 'allOptionsContainer',
    selectedOptionSection: 'selectedOptionSection',
    selectedOptionDetails: 'selectedOptionDetails',
    detailed: 'detailedRecommendations',
    recommendationsSection: 'recommendationsSection'
  };

  const missingElements = [];
  for (const [varName, elementId] of Object.entries(requiredElements)) {
    const element = document.getElementById(elementId);
    if (!element) {
      missingElements.push(elementId);
    }
  }

  if (missingElements.length > 0) {
    console.error('Missing required DOM elements:', missingElements);
    return false;
  }
  return true;
}

let chartInstance = null;
let currentOptions = [];
let selectedOptionIndex = 0;

// Global error handler
window.addEventListener('error', function(event) {
  console.error('Global error caught:', event.error);
  console.error('Error details:', {
    message: event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno
  });
});

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, initializing application...');

  if (!validateRequiredElements()) {
    console.error('Required elements missing during initialization');
    alert('Page not loaded correctly. Please refresh the page.');
    return;
  }

  console.log('Application initialized successfully');
});

if (form) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Validate DOM elements before proceeding
    if (!validateRequiredElements()) {
      alert('Page not loaded correctly. Please refresh the page.');
      return;
    }

    // Show loading state
    const submitBtn = e.target.querySelector('button[type="submit"]');
    if (!submitBtn) {
      console.error('Submit button not found');
      return;
    }

    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Getting Recommendations...';
    submitBtn.disabled = true;

    try {
      const payload = getFormData();
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();

      // Validate response structure
      if (!data.options || !Array.isArray(data.options) || data.options.length === 0) {
        throw new Error('Invalid response: No investment options received');
      }

      displayResults(data);
    } catch (error) {
      console.error('Error:', error);
      alert(`Sorry, there was an error getting your recommendations: ${error.message}. Please try again.`);
    } finally {
      submitBtn.textContent = originalText;
      submitBtn.disabled = false;
    }
  });
} else {
  console.error('Form element not found');
}

function getFormData() {
  try {
    const amountEl = document.getElementById('amount');
    const horizonEl = document.getElementById('horizon');
    const riskEl = document.getElementById('risk_tolerance');
    const goalEl = document.getElementById('financial_goal');
    const ageEl = document.getElementById('age');

    if (!amountEl || !horizonEl || !riskEl || !goalEl || !ageEl) {
      throw new Error('Form elements not found');
    }

    return {
      amount: parseFloat(amountEl.value || '0'),
      horizon: parseInt(horizonEl.value || '0', 10),
      risk_tolerance: riskEl.value,
      financial_goal: goalEl.value,
      age: parseInt(ageEl.value || '18', 10)
    };
  } catch (error) {
    console.error('Error getting form data:', error);
    throw new Error('Unable to read form data');
  }
}

function displayResults(response) {
  currentOptions = response.options || [];

  if (currentOptions.length === 0) {
    alert('No investment options received. Please try again.');
    return;
  }

  // Show sections with null checks
  if (optionsSection) optionsSection.style.display = 'block';
  if (selectedOptionSection) selectedOptionSection.style.display = 'block';
  if (recommendationsSection) recommendationsSection.style.display = 'block';

  // Display all options simultaneously
  displayAllOptions();

  // Display first option as selected by default
  selectedOptionIndex = 0;
  displaySelectedOption(0);

  // Scroll to results
  if (optionsSection) {
    optionsSection.scrollIntoView({ behavior: 'smooth' });
  }
}

function displayAllOptions() {
  if (!allOptionsContainer) {
    console.error('allOptionsContainer element not found');
    return;
  }

  allOptionsContainer.innerHTML = '';

  currentOptions.forEach((option, index) => {
    const optionCard = document.createElement('div');
    optionCard.className = `option-card ${index === selectedOptionIndex ? 'selected' : ''}`;
    optionCard.style.cssText = `
      border: 2px solid ${index === selectedOptionIndex ? 'var(--success)' : 'var(--border)'};
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
      background: ${index === selectedOptionIndex ? 'var(--light-success)' : 'var(--card-bg)'};
      cursor: pointer;
      transition: all 0.3s ease;
    `;

    const recommendations = option.recommendations || [];
    const fundNames = recommendations.slice(0, 5).map(fund => fund.name || fund.scheme_name || fund.fund_name || 'Fund').join(', ');
    const moreCount = recommendations.length > 5 ? recommendations.length - 5 : 0;

    optionCard.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 15px;">
        <!-- Header -->
        <div style="display: flex; align-items: center; gap: 10px;">
          <h3 style="margin: 0; color: var(--primary);">Option ${String.fromCharCode(65 + index)}: ${option.option_name}</h3>
          ${index === selectedOptionIndex ? '<span style="background: var(--success); color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">✓ Selected</span>' : ''}
        </div>

        <!-- Chart -->
        <div style="width: 100%; height: 280px; margin-bottom: 10px;">
          <canvas id="chart${index}" style="width: 100%; height: 100%;"></canvas>
        </div>

        <!-- Description -->
        <p style="color: var(--secondary); margin-bottom: 15px;">${option.option_description}</p>

        <!-- Stats Grid -->
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 15px;">
          <div style="text-align: center; padding: 8px; background: var(--light-bg); border-radius: 6px;">
            <div style="font-size: 12px; color: var(--secondary);">Expected Returns</div>
            <div style="font-weight: bold; color: var(--accent); font-size: 14px;">${option.expected_return_min}% - ${option.expected_return_max}%</div>
          </div>
          <div style="text-align: center; padding: 8px; background: var(--light-bg); border-radius: 6px;">
            <div style="font-size: 12px; color: var(--secondary);">Risk Level</div>
            <div style="font-weight: bold; color: var(--primary); font-size: 14px;">${option.risk_level}</div>
          </div>
          <div style="text-align: center; padding: 8px; background: var(--light-bg); border-radius: 6px;">
            <div style="font-size: 12px; color: var(--secondary);">Total Amount</div>
            <div style="font-weight: bold; color: var(--success); font-size: 14px;">₹${option.total_investment.toLocaleString()}</div>
          </div>
          <div style="text-align: center; padding: 8px; background: var(--light-bg); border-radius: 6px;">
            <div style="font-size: 12px; color: var(--secondary);">Fund Count</div>
            <div style="font-weight: bold; color: var(--accent); font-size: 14px;">${recommendations.length} Funds</div>
          </div>
        </div>

        <!-- Funds List -->
        <div style="margin-bottom: 15px;">
          <h4 style="margin: 0 0 8px 0; color: var(--primary); font-size: 14px;">📈 Funds in this option:</h4>
          <p style="font-size: 13px; color: var(--secondary); line-height: 1.4; margin: 0;">
            ${fundNames}${moreCount > 0 ? ` and ${moreCount} more funds` : ''}
          </p>
        </div>

        <!-- Selection Button -->
        <button class="select-option-btn ${index === selectedOptionIndex ? 'selected' : ''}"
                onclick="selectOptionFromAll(${index})"
                style="width: 100%; padding: 12px; border: none; border-radius: 6px;
                       background: ${index === selectedOptionIndex ? 'var(--success)' : 'var(--primary)'};
                       color: white; font-weight: bold; cursor: pointer; font-size: 14px;">
          ${index === selectedOptionIndex ? '✓ Currently Selected' : 'Select This Option'}
        </button>
      </div>
    `;

    allOptionsContainer.appendChild(optionCard);

    // Create pie chart for this option
    setTimeout(() => createOptionChart(option, index), 100);
  });
}

function createOptionChart(option, index) {
  try {
    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
      console.error('Chart.js library not loaded');
      return;
    }

    const canvas = document.getElementById(`chart${index}`);
    if (!canvas) {
      console.error(`Canvas element chart${index} not found`);
      return;
    }

    if (!option || !option.recommendations) {
      console.error('Invalid option data for chart creation');
      return;
    }

    const recommendations = option.recommendations || [];
    if (recommendations.length === 0) {
      console.warn('No recommendations found for chart creation');
      return;
    }

    const labels = recommendations.map(r => {
      const name = r.name || r.scheme_name || r.fund_name || 'Unknown Fund';
      return name.length > 20 ? name.substring(0, 20) + '...' : name;
    });

    const data = recommendations.map(r => r.allocation || 0);

    new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor: [
            '#0E7C7B', '#1E90FF', '#F59E0B', '#10B981', '#EF4444', '#8B5CF6',
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'
          ],
          borderWidth: 2,
          borderColor: '#FFFFFF'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: `${option.option_name || 'Investment Option'} - Portfolio Allocation`,
            font: { size: 14, weight: 'bold' },
            color: '#0F172A'
          },
          legend: {
            position: 'bottom',
            labels: {
              padding: 8,
              usePointStyle: true,
              font: { size: 10 }
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.label || '';
                const value = context.parsed || 0;
                return `${label}: ${value}%`;
              }
            }
          }
        }
      }
    });
  } catch (error) {
    console.error(`Error creating chart for option ${index}:`, error);
  }
}

function selectOptionFromAll(optionIndex) {
  try {
    if (!currentOptions[optionIndex]) {
      console.error('Invalid option index:', optionIndex);
      return;
    }

    selectedOptionIndex = optionIndex;

    // Update all option cards styling
    const optionCards = document.querySelectorAll('.option-card');
    optionCards.forEach((card, index) => {
      if (!card) return;
      if (index === optionIndex) {
        card.style.border = '2px solid var(--success)';
        card.style.background = 'var(--light-success)';
        card.classList.add('selected');
      } else {
        card.style.border = '2px solid var(--border)';
        card.style.background = 'var(--card-bg)';
        card.classList.remove('selected');
      }
    });

    // Update button texts and selected indicators
    const buttons = document.querySelectorAll('.select-option-btn');
    buttons.forEach((btn, index) => {
      if (!btn) return;
      if (index === optionIndex) {
        btn.textContent = '✓ Currently Selected';
        btn.style.background = 'var(--success)';
        btn.classList.add('selected');
      } else {
        btn.textContent = 'Select This Option';
        btn.style.background = 'var(--primary)';
        btn.classList.remove('selected');
      }
    });

    // Update selected indicators
    document.querySelectorAll('.option-card h3').forEach((title, index) => {
      if (!title) return;
      const existingSpan = title.querySelector('span');
      if (existingSpan) existingSpan.remove();

      if (index === optionIndex) {
        const span = document.createElement('span');
        span.style.cssText = 'background: var(--success); color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px;';
        span.textContent = '✓ Selected';
        title.appendChild(span);
      }
    });

    // Display detailed info for selected option
    displaySelectedOption(optionIndex);

    // Show success message
    const option = currentOptions[optionIndex];
    if (option && option.option_name) {
      alert(`Great choice! You've selected "${option.option_name}". Review the detailed breakdown below.`);
    }
  } catch (error) {
    console.error('Error selecting option:', error);
    alert('There was an error selecting this option. Please try again.');
  }
}

function displaySelectedOption(optionIndex) {
  if (!currentOptions[optionIndex]) return;

  if (!selectedOptionDetails) {
    console.error('selectedOptionDetails element not found');
    return;
  }

  const option = currentOptions[optionIndex];
  const recommendations = option.recommendations || [];
  const formData = getFormData();

  selectedOptionDetails.innerHTML = `
    <div style="background: var(--light-success); padding: 20px; border-radius: 8px; border-left: 4px solid var(--success); margin-bottom: 20px;">
      <h3 style="margin: 0 0 10px 0; color: var(--success);">✓ Selected: ${option.option_name}</h3>
      <p style="margin: 0; color: var(--secondary);">${option.option_description}</p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px;">
      <div style="background: var(--card-bg); padding: 16px; border-radius: 8px; text-align: center;">
        <div style="color: var(--secondary); font-size: 14px;">Total Investment</div>
        <div style="color: var(--success); font-size: 20px; font-weight: bold;">₹${option.total_investment.toLocaleString()}</div>
      </div>
      <div style="background: var(--card-bg); padding: 16px; border-radius: 8px; text-align: center;">
        <div style="color: var(--secondary); font-size: 14px;">Expected Returns</div>
        <div style="color: var(--accent); font-size: 20px; font-weight: bold;">${option.expected_return_min}% - ${option.expected_return_max}%</div>
      </div>
      <div style="background: var(--card-bg); padding: 16px; border-radius: 8px; text-align: center;">
        <div style="color: var(--secondary); font-size: 14px;">Risk Level</div>
        <div style="color: var(--primary); font-size: 20px; font-weight: bold;">${option.risk_level}</div>
      </div>
      <div style="background: var(--card-bg); padding: 16px; border-radius: 8px; text-align: center;">
        <div style="color: var(--secondary); font-size: 14px;">Investment Period</div>
        <div style="color: var(--primary); font-size: 20px; font-weight: bold;">${formData.horizon} Years</div>
      </div>
    </div>

    <h3>📋 Complete Fund Breakdown</h3>
    <div style="display: grid; gap: 12px; margin-bottom: 20px;">
      ${recommendations.map(fund => `
        <div style="background: var(--card-bg); padding: 16px; border-radius: 8px; border: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center;">
          <div>
            <strong style="color: var(--primary);">${fund.name}</strong><br>
            <small style="color: var(--secondary);">${fund.category} • ${fund.type}</small>
          </div>
          <div style="text-align: right;">
            <strong style="color: var(--accent); font-size: 18px;">${fund.allocation}%</strong><br>
            <small style="color: var(--secondary);">₹${fund.amount.toLocaleString()}</small>
          </div>
        </div>
      `).join('')}
    </div>
  `;

  // Update detailed strategy section
  displayDetailedStrategy(option);
}


function displayDetailedStrategy(option) {
  const currentSelectedOption = currentOptions[selectedOptionIndex] || option;

  // Create beginner-friendly content
  const beginnerContent = createBeginnerFriendlyContent(option);

  // Ensure detailed element exists
  if (!detailed) {
    console.error('Detailed recommendations element not found');
    return;
  }

  detailed.innerHTML = `
    <div class="strategy-content">
      <h3>💡 What This Investment Plan Means for You</h3>
      ${beginnerContent}

      <h3>🎯 Next Steps</h3>
      <ul>
        <li><strong>Review:</strong> Scroll up to compare all ${currentOptions.length} options and confirm your selection</li>
        <li><strong>Research:</strong> Learn more about each fund's past performance and ratings</li>
        <li><strong>KYC:</strong> Complete your KYC if not done already (PAN + Aadhaar)</li>
        <li><strong>Start Investment:</strong> Open account with recommended mutual fund platforms</li>
        <li><strong>Setup SIP:</strong> Automate monthly investments for discipline</li>
        <li><strong>Monitor:</strong> Review your investments every 6 months</li>
        <li><strong>Rebalance:</strong> Adjust allocations annually to maintain target percentages</li>
        <li><strong>Consult:</strong> Consider speaking with a SEBI-registered financial advisor</li>
      </ul>

      <h3>💼 Recommended Investment Platforms</h3>
      <p>Use direct mutual fund platforms to avoid commission fees (saves 0.5-1% annually):</p>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
        <div style="background: var(--light-bg); padding: 12px; border-radius: 6px;">
          <strong>Groww</strong><br>
          <small>User-friendly interface, good for beginners</small>
        </div>
        <div style="background: var(--light-bg); padding: 12px; border-radius: 6px;">
          <strong>Zerodha Coin</strong><br>
          <small>Part of Zerodha ecosystem, low fees</small>
        </div>
        <div style="background: var(--light-bg); padding: 12px; border-radius: 6px;">
          <strong>Paytm Money</strong><br>
          <small>Easy UPI payments and tracking</small>
        </div>
        <div style="background: var(--light-bg); padding: 12px; border-radius: 6px;">
          <strong>ET Money</strong><br>
          <small>Comprehensive financial tracking</small>
        </div>
      </div>

      <div style="background: var(--warning); color: white; padding: 16px; border-radius: 8px; margin-top: 20px;">
        <strong>⚠️ Important Disclaimer:</strong> Mutual fund investments are subject to market risks. Past performance does not guarantee future returns. Please read scheme documents carefully before investing.
      </div>
    </div>
  `;
}


function cleanMarkdownText(text) {
  if (!text) return '';

  return text
    // Remove markdown bold formatting (**text**)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Remove single asterisk formatting (*text*)
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    // Convert line breaks to proper paragraphs
    .split('\n\n')
    .filter(paragraph => paragraph.trim())
    .map(paragraph => `<p>${paragraph.trim()}</p>`)
    .join('')
    // Handle remaining single line breaks
    .replace(/\n/g, '<br>');
}

function createBeginnerFriendlyContent(option) {
  try {
    const riskLevel = option?.risk_level || 'Moderate';
    const totalInvestment = option?.total_investment || 0;
    const formData = getFormData();
    const horizon = formData?.horizon || 1;

  let riskExplanation = '';
  let expectedOutcome = '';

  if (riskLevel.toLowerCase().includes('low')) {
    riskExplanation = 'You prefer safety over high returns. Your money will be mostly in stable investments like bonds and debt funds.';
    expectedOutcome = 'Your investments should be quite safe, but returns will be moderate.';
  } else if (riskLevel.toLowerCase().includes('high')) {
    riskExplanation = 'You\'re willing to take more risk for potentially higher returns. Your money will be mostly in stock market investments.';
    expectedOutcome = 'Your investments may go up and down more, but could give higher returns over time.';
  } else {
    riskExplanation = 'You want a balanced approach. Your money will be split between safe and growth investments.';
    expectedOutcome = 'Your investments will have moderate risk with good potential for growth.';
  }

  return `
    <p><strong>Investment Amount:</strong> ₹${totalInvestment.toLocaleString()} over ${horizon} years</p>
    <p><strong>Your Risk Profile:</strong> ${riskExplanation}</p>
    <p><strong>What to Expect:</strong> ${expectedOutcome}</p>
    <p><strong>This Option:</strong> ${option?.option_description || 'Investment strategy tailored to your profile'}</p>

    <h4>📚 Simple Explanations:</h4>
    <ul>
      <li><strong>Mutual Fund:</strong> A basket where many people pool money to buy stocks/bonds</li>
      <li><strong>Allocation:</strong> How much percentage of your money goes to each fund</li>
      <li><strong>Expected Return:</strong> How much profit you might make per year (not guaranteed)</li>
      <li><strong>Equity Funds:</strong> Invest in company stocks (higher risk, higher potential returns)</li>
      <li><strong>Debt Funds:</strong> Invest in bonds/loans (lower risk, steady returns)</li>
      <li><strong>Hybrid Funds:</strong> Mix of both stocks and bonds (balanced risk)</li>
    </ul>
  `;
  } catch (error) {
    console.error('Error creating beginner content:', error);
    return '<p>Investment strategy details will be displayed here.</p>';
  }
}
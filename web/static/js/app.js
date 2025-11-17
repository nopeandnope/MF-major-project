document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('investmentForm');
    const portfolioContent = document.getElementById('portfolioContent');
    const recommendationsSection = document.getElementById('recommendationsSection');
    const detailedRecommendations = document.getElementById('detailedRecommendations');
    let chartInstance = null;
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = {
            amount: parseFloat(document.getElementById('amount').value),
            horizon: parseInt(document.getElementById('horizon').value),
            risk_tolerance: document.getElementById('riskTolerance').value,
            financial_goal: document.getElementById('financialGoal').value,
            age: parseInt(document.getElementById('age').value)
        };
        
        portfolioContent.innerHTML = `
            <div class="empty-state">
                <div class="spinner"></div>
                <p>Analyzing your profile...</p>
            </div>
        `;
        
        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            displayPortfolio(data, formData);
            displayDetailedRecommendations(data, formData);
            
        } catch (error) {
            console.error('Error:', error);
            portfolioContent.innerHTML = `
                <div class="empty-state">
                    <p style="color: var(--text-primary);">Unable to generate recommendations. Please try again.</p>
                </div>
            `;
        }
    });
    
    function displayPortfolio(data, formData) {
        let html = `
            <div class="portfolio-summary">
                <div class="summary-header">
                    <div class="summary-item">
                        <p>Total Investment</p>
                        <p>₹${formData.amount.toLocaleString()}</p>
                    </div>
                    <div class="summary-item">
                        <p>Expected Returns</p>
                        <p>${data.expected_return_min}-${data.expected_return_max}%</p>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="portfolioChart"></canvas>
                </div>
                <div class="fund-list">
        `;
        
        data.recommendations.forEach(fund => {
            html += `
                <div class="fund-item">
                    <div class="fund-info">
                        <h4>${fund.name}</h4>
                        <p>${fund.category} • ${fund.type}</p>
                    </div>
                    <div class="fund-allocation">
                        <div class="allocation-percent">${fund.allocation}%</div>
                        <div class="allocation-amount">₹${Math.round(fund.amount).toLocaleString()}</div>
                    </div>
                </div>
            `;
        });
        
        html += '</div></div>';
        portfolioContent.innerHTML = html;
        
        drawChart(data.recommendations);
    }
    
    function drawChart(recommendations) {
        const ctx = document.getElementById('portfolioChart').getContext('2d');
        
        if (chartInstance) {
            chartInstance.destroy();
        }
        
        const colors = [
            '#1a1a2e',
            '#16213e', 
            '#0f3460',
            '#533483',
            '#e94560',
            '#f5f5f7'
        ];
        
        chartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: recommendations.map(f => f.name),
                datasets: [{
                    data: recommendations.map(f => f.allocation),
                    backgroundColor: colors,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            font: {
                                size: 11,
                                family: 'Inter'
                            },
                            color: '#533483'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    function displayDetailedRecommendations(data, formData) {
        recommendationsSection.style.display = 'block';
        
        const cleanStrategy = data.strategy
            .replace(/\*\*/g, '')
            .replace(/\*/g, '•')
            .replace(/#+/g, '');
        
        const sections = cleanStrategy.split('\n\n');
        let html = '<div class="strategy-content">';
        
        sections.forEach(section => {
            if (section.trim()) {
                html += '<div class="strategy-section">';
                const lines = section.split('\n');
                
                if (lines[0].includes('PORTFOLIO') || lines[0].includes('STRATEGY')) {
                    html += `<h3>${lines[0]}</h3>`;
                    html += '<ul>';
                    for (let i = 1; i < lines.length; i++) {
                        if (lines[i].trim()) {
                            html += `<li>${lines[i].replace(/[•→]/g, '').trim()}</li>`;
                        }
                    }
                    html += '</ul>';
                } else {
                    html += `<p>${section}</p>`;
                }
                html += '</div>';
            }
        });
        
        html += '</div>';
        detailedRecommendations.innerHTML = html;
    }
});

const style = document.createElement('style');
style.textContent = `
    .spinner {
        border: 2px solid var(--border);
        border-top: 2px solid var(--primary);
        border-radius: 50%;
        width: 32px;
        height: 32px;
        animation: spin 1s linear infinite;
        margin: 0 auto 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);
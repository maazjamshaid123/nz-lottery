import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import sys
import io

# Import our NZ Powerball optimizer with error handling
try:
    from main import NZPowerballTicketOptimizer, LSTM_AVAILABLE
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import main module: {e}")
    IMPORT_SUCCESS = False
    LSTM_AVAILABLE = False
    NZPowerballTicketOptimizer = None

# Page configuration
st.set_page_config(
    page_title="NZ Powerball Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .ticket-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #e9ecef;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Set display mode to Table View only
    display_mode = "📋 Table View (Compact)"

    # Check imports first
    if not IMPORT_SUCCESS:
        st.error("❌ Failed to import required modules. Please check your installation.")
        st.info("Run: pip install -r requirements_streamlit.txt")
        return

    # Main header
    st.markdown('<div class="main-header">🎯 NZ Powerball Ticket Optimizer</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    🎲 <strong>Smart Lottery Optimization with BiLSTM AI</strong><br>
    Generate optimized lottery tickets using advanced bidirectional LSTM neural networks.
    Upload your historical data and compare with official draws to see AI accuracy!
    </div>
    """, unsafe_allow_html=True)

    # Official Draw Comparison Section
    st.subheader("🎯 Official Draw Comparison (Optional)")
    st.markdown("Enter the official draw results to see how our AI predictions compare!")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        official_numbers = st.text_input(
            "Official Main Numbers (6 numbers, 1-40)",
            placeholder="e.g., 5, 12, 18, 23, 31, 39",
            help="Enter the 6 main numbers from the official draw, separated by commas"
        )
    with col2:
        official_bonus = st.number_input(
            "Official Bonus Ball (1-40)",
            min_value=1,
            max_value=40,
            value=1,
            help="Enter the official bonus ball number"
        )
    with col3:
        official_powerball = st.number_input(
            "Official Powerball (1-10)",
            min_value=1,
            max_value=10,
            value=1,
            help="Enter the official powerball number"
        )
    
    # Parse official numbers
    official_main_list = []
    if official_numbers:
        try:
            official_main_list = [int(x.strip()) for x in official_numbers.split(',') if x.strip()]
            if len(official_main_list) != 6:
                st.warning("⚠️ Please enter exactly 6 main numbers")
                official_main_list = []
            elif not all(1 <= num <= 40 for num in official_main_list):
                st.warning("⚠️ All main numbers must be between 1 and 40")
                official_main_list = []
            elif len(set(official_main_list)) != 6:
                st.warning("⚠️ All main numbers must be unique")
                official_main_list = []
            else:
                st.success(f"✅ Official draw: {sorted(official_main_list)} + Bonus: {official_bonus} + PB: {official_powerball}")
        except ValueError:
            st.warning("⚠️ Please enter valid numbers separated by commas")
            official_main_list = []

    st.markdown("---")

    # Simplified Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File upload
        st.subheader("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file with historical draws",
            type=['csv'],
            help="Upload your NZ Powerball historical data"
        )

        # Simple prediction toggle
        st.subheader("🎯 Prediction Mode")
        if LSTM_AVAILABLE:
            use_tft = st.toggle(
                "🤖 Enable AI Predictions",
                value=True,
                help="Use AI to find patterns in historical data"
            )
        else:
            use_tft = False
            st.error("❌ AI requires PyTorch installation")

        # Simple ticket count
        st.subheader("🎫 Tickets")
        num_tickets = st.slider(
            "Number of tickets",
            min_value=1,
            max_value=15,
            value=6,
            help="More tickets = better coverage"
        )
        
        ticket_cost = num_tickets * 1.50
        st.info(f"💰 Cost: ${ticket_cost:.2f} NZD")

        # Generate button
        st.markdown("---")
        generate_button = st.button(
            "🚀 Generate Tickets!",
            type="primary",
            use_container_width=True
        )

    # Set optimal backend defaults
    tft_mode = "probabilistic"  # Best performance
    custom_seed = None  # Random for each generation
    jackpot_amount = 50000000  # Standard jackpot
    estimated_sales = 400000  # Typical sales

    # Display mode is fixed to Table View (Compact) only

    # Main content area
    if uploaded_file is not None:
        # Display file info
        st.subheader("📊 Dataset Preview")

        # Read the uploaded file
        try:
            df = pd.read_csv(uploaded_file)

            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Draws", len(df))
            with col2:
                st.metric("Data Columns", len(df.columns))
            with col3:
                # Remove date range display - not necessary for end users
                st.metric("Data Quality", "Valid")

            # Show sample data
            st.dataframe(df.head(), use_container_width=True)

            # Check for required columns
            required_cols = ['1', '2', '3', '4', '5', '6', 'Power Ball']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: 1, 2, 3, 4, 5, 6, Power Ball, Bonus Ball (optional)")
                return

            # Success message
            st.success("✅ Dataset loaded successfully!")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return

        # Generate button action
        if generate_button:
            generate_tickets(df, use_tft, tft_mode, num_tickets, custom_seed, jackpot_amount, estimated_sales, official_main_list, official_bonus, official_powerball)

    else:
        # Welcome screen
        st.info("👆 Please upload your NZ Powerball historical data to get started!")

        # Sample data format
        st.subheader("📋 Expected Data Format")

        # Download sample data
        sample_csv = """Draw,Draw Date,1,2,3,4,5,6,Bonus Ball,Power Ball
2511,Wednesday 27 August 2025,9,12,13,14,21,32,28,2
2510,Saturday 23 August 2025,13,16,25,17,31,24,38,8
2509,Wednesday 20 August 2025,16,11,7,37,12,27,26,9
2508,Saturday 16 August 2025,33,39,38,28,34,6,40,10
2507,Wednesday 13 August 2025,10,9,15,22,27,35,25,10
2506,Saturday 09 August 2025,19,21,11,35,6,33,27,3
2505,Wednesday 06 August 2025,10,14,12,20,1,11,24,10
2504,Saturday 02 August 2025,19,29,28,40,31,3,2,9
2503,Wednesday 30 July 2025,33,5,10,27,38,28,40,6
2502,Saturday 26 July 2025,17,19,23,26,32,40,14,7
2501,Wednesday 23 July 2025,8,9,18,24,25,39,11,1
2500,Saturday 19 July 2025,6,7,14,20,29,37,15,5
2499,Wednesday 16 July 2025,4,12,21,28,31,34,22,3
2498,Saturday 12 July 2025,2,8,13,16,30,36,17,8
2497,Wednesday 09 July 2025,1,5,10,23,35,38,19,6
2496,Saturday 05 July 2025,3,11,18,27,32,39,20,4
2495,Wednesday 02 July 2025,7,15,22,26,33,40,12,9
2494,Saturday 28 June 2025,9,17,24,29,36,37,13,2
2493,Wednesday 25 June 2025,4,14,19,25,30,38,21,7
2492,Saturday 21 June 2025,6,12,20,28,34,39,16,5"""

        st.download_button(
            label="📥 Download Sample Data (20 draws)",
            data=sample_csv,
            file_name="nz_powerball_sample.csv",
            mime="text/csv",
            help="Download sample NZ Powerball data to try the app"
        )

        st.dataframe(pd.read_csv(io.StringIO(sample_csv)).head(), use_container_width=True)

        st.info("💡 **Tip**: Start with this sample data to see how the app works!")

def generate_tickets(df, use_tft, tft_mode, num_tickets, custom_seed, jackpot_amount, estimated_sales, official_main=None, official_bonus=None, official_powerball=None):
    """Generate optimized tickets using the uploaded data."""

    # Create temporary file for the dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
        temp_csv_path = temp_file.name

    try:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Initializing optimizer...")
        progress_bar.progress(10)

        # Initialize optimizer
        seed = custom_seed if custom_seed else None
        optimizer = NZPowerballTicketOptimizer(temp_csv_path, seed=seed, use_tft=use_tft)

        status_text.text("📊 Loading historical data...")
        progress_bar.progress(30)

        # Load data (this will train TFT models if enabled)
        optimizer.load_historical_data()

        # Surface engine status in Streamlit
        if use_tft:
            st.success("🤖 AI mode ON (BiLSTM)")
        else:
            st.warning("🎲 Uniform mode (no BiLSTM)")

        # Set TFT mode if using TFT
        if use_tft:
            optimizer.tft_mode = tft_mode

        status_text.text("🎯 Generating optimized tickets...")
        progress_bar.progress(60)

        # Generate portfolio with optimal constraints
        portfolio = optimizer.generate_ticket_portfolio(
            num_tickets=num_tickets,
            diversity_constraints={
                'max_consecutive': 2,  # Optimal for lottery
                'balance_odd_even': True,
                'avoid_popular_patterns': True,
                'sum_range': (85, 165),  # Statistical sweet spot
                'digit_ending_variety': True
            }
        )

        status_text.text("📈 Calculating expected value...")
        progress_bar.progress(80)

        # Calculate expected value
        expected_value_analysis = optimizer.calculate_expected_value(
            portfolio,
            jackpot_amount=jackpot_amount,
            estimated_sales=estimated_sales,
            prize_structure=None
        )

        progress_bar.progress(100)
        status_text.text("✅ Optimization complete!")

        # Display results
        display_results(optimizer, portfolio, expected_value_analysis, use_tft, tft_mode, "📋 Table View (Compact)", official_main, official_bonus, official_powerball)

    except Exception as e:
        st.error(f"❌ Error during optimization: {str(e)}")
        st.info("💡 Try with simpler settings or check your data format")

    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_csv_path)
        except:
            pass

def display_results(optimizer, portfolio, expected_value_analysis, use_tft, tft_mode, display_mode, official_main=None, official_bonus=None, official_powerball=None):
    """Display the optimization results in a user-friendly format."""

    st.markdown('<div class="sub-header">🎉 Your Optimized Tickets</div>', unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Tickets", len(portfolio))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        total_investment = len(portfolio) * 1.50
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Investment", f"${total_investment:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        expected_value = expected_value_analysis['expected_value']
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Expected Value", f"${expected_value:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        roi = expected_value_analysis['return_on_investment'] * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("ROI", f"{roi:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Tickets display with beautiful ball styling
    st.subheader("🎫 Your Optimized Tickets")

    # Display mode is fixed to Table View (Compact) for easy comparison
    st.info("📋 **Table View**: All tickets displayed in a compact, comparable format for easy analysis")

    # Display tickets in Table View (Compact) only
    display_tickets_table(portfolio, official_main, official_bonus, official_powerball)

    # Enhanced Portfolio Analysis with beautiful visualizations
    st.subheader("📊 Portfolio Analysis")

    # Add CSS for enhanced analysis cards (Light Mode)
    st.markdown("""
    <style>
        .analysis-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 2px solid #dee2e6;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        }
        .analysis-header {
            color: #1f77b4;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-align: center;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            padding: 0.75rem;
            background: #f8f9fa;
            border-radius: 10px;
            transition: all 0.3s ease;
            border: 1px solid #e9ecef;
        }
        .metric-row:hover {
            background: #e9ecef;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-label {
            color: #495057;
            font-weight: 600;
        }
        .metric-value {
            color: #1f77b4;
            font-weight: 700;
            font-size: 1.1rem;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 0.5rem;
        }
        .status-good { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-excellent { background-color: #007bff; }
    </style>
    """, unsafe_allow_html=True)

    # Calculate analysis metrics
    sums = [t['sum'] for t in portfolio]
    odd_counts = [t['odd_count'] for t in portfolio]
    all_digits = []
    for t in portfolio:
        all_digits.extend(t['last_digits'])
    unique_digits = len(set(all_digits))

    # Sum Distribution Analysis
    sum_range = max(sums) - min(sums)
    target_range = 180 - 70  # 70-180 is target
    sum_coverage = min(100, (sum_range / target_range) * 100)

    # Odd/Even Analysis
    avg_odd = sum(odd_counts) / len(odd_counts)
    odd_even_balance = 1 - abs(avg_odd - 3) / 3  # 3 is ideal

    # Last Digit Variety
    digit_coverage = (unique_digits / 10) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="analysis-card">
            <div class="analysis-header">📈 Sum Distribution</div>
        </div>
        """, unsafe_allow_html=True)

        # Sum range metric
        sum_status = "excellent" if sum_coverage > 80 else "good" if sum_coverage > 60 else "warning"
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-label">Sum Range:</span>
            <span class="metric-value">{min(sums)} - {max(sums)} <span class="status-indicator status-{sum_status}"></span></span>
        </div>
        """, unsafe_allow_html=True)

        # Unique sums metric
        unique_sum_percentage = (len(set(sums)) / len(portfolio)) * 100
        unique_status = "excellent" if unique_sum_percentage > 80 else "good" if unique_sum_percentage > 60 else "warning"
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-label">Unique Sums:</span>
            <span class="metric-value">{len(set(sums))}/{len(portfolio)} ({unique_sum_percentage:.1f}%) <span class="status-indicator status-{unique_status}"></span></span>
        </div>
        """, unsafe_allow_html=True)

        # Target achievement
        target_status = "good" if 70 <= min(sums) and max(sums) <= 180 else "warning"
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-label">Target Range (70-180):</span>
            <span class="metric-value">{"✓ Achieved" if target_status == "good" else "⚠ Outside Range"} <span class="status-indicator status-{target_status}"></span></span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="analysis-card">
            <div class="analysis-header">⚖️ Balance Analysis</div>
        </div>
        """, unsafe_allow_html=True)

        # Odd/Even balance
        balance_percentage = odd_even_balance * 100
        balance_status = "excellent" if balance_percentage > 80 else "good" if balance_percentage > 60 else "warning"
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-label">Odd/Even Balance:</span>
            <span class="metric-value">{min(odd_counts)} - {max(odd_counts)} (avg: {avg_odd:.1f}) <span class="status-indicator status-{balance_status}"></span></span>
        </div>
        """, unsafe_allow_html=True)

        # Last digit variety
        digit_status = "excellent" if digit_coverage > 80 else "good" if digit_coverage > 60 else "warning"
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-label">Last Digit Variety:</span>
            <span class="metric-value">{unique_digits}/10 ({digit_coverage:.1f}%) <span class="status-indicator status-{digit_status}"></span></span>
        </div>
        """, unsafe_allow_html=True)

        # Overall portfolio health
        overall_score = (sum_coverage + unique_sum_percentage + balance_percentage + digit_coverage) / 4
        overall_status = "excellent" if overall_score > 80 else "good" if overall_score > 60 else "warning"
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-label">Portfolio Health:</span>
            <span class="metric-value">{overall_score:.1f}% <span class="status-indicator status-{overall_status}"></span></span>
        </div>
        """, unsafe_allow_html=True)

    # Add a summary insight
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fff3cd, #fce4a3); border-radius: 15px; padding: 1.5rem; margin-top: 2rem; border: 2px solid #ffc107;">
        <h4 style="color: #856404; text-align: center; margin-bottom: 1rem;">💡 Portfolio Insights</h4>
        <div style="color: #495057; text-align: center;">
    """, unsafe_allow_html=True)

    if overall_score > 80:
        st.markdown("🌟 **Excellent Diversity!** Your portfolio shows great variety across all metrics. This maximizes your coverage!")
    elif overall_score > 60:
        st.markdown("👍 **Good Balance!** Your tickets have decent diversity. Consider adding more variety for better coverage.")
    else:
        st.markdown("⚠️ **Limited Diversity** Your tickets are quite similar. Try increasing the number of tickets for better spread.")

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Enhanced Winning Probabilities section
    st.subheader("🎯 Winning Probabilities")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 20px; padding: 1.5rem; margin: 1rem 0; border: 2px solid #dee2e6;">
        <h4 style="color: #1f77b4; text-align: center; margin-bottom: 1rem;">💰 Prize Division Odds</h4>
    </div>
    """, unsafe_allow_html=True)

    # Display top 5 prize divisions with enhanced styling
    prize_divisions = list(expected_value_analysis['division_breakdown'].items())[:5]

    for division, details in prize_divisions:
        prob_pct = details['probability'] * 100
        if prob_pct > 0.001:  # Only show meaningful probabilities
            prize_amount = details.get('prize', 0)
            expected_prize = details.get('expected_prize', 0)

            # Determine probability level for color coding
            if prob_pct > 1:
                prob_color = "#28a745"  # Green for good odds
                prob_icon = "🎯"
            elif prob_pct > 0.1:
                prob_color = "#ffc107"  # Yellow for medium odds
                prob_icon = "🎲"
            else:
                prob_color = "#dc3545"  # Red for low odds
                prob_icon = "🎪"

            st.markdown(f"""
            <div style="background: #ffffff; border-radius: 15px; padding: 1rem; margin: 0.5rem 0; border: 2px solid #dee2e6; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.08);" onmouseover="this.style.background='#f8f9fa'; this.style.boxShadow='0 6px 20px rgba(0,0,0,0.12)'" onmouseout="this.style.background='#ffffff'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.08)'">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <strong style="color: #1f77b4; font-size: 1.1rem;">{prob_icon} {division}</strong>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div style="color: {prob_color}; font-weight: 700; font-size: 1.2rem;">{prob_pct:.4f}%</div>
                        <div style="color: #6c757d; font-size: 0.9rem;">Probability</div>
                    </div>
                    <div style="flex: 1; text-align: right;">
                        <div style="color: #28a745; font-weight: 700;">${prize_amount:,.0f}</div>
                        <div style="color: #6c757d; font-size: 0.9rem;">Prize</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Educational results explanation
    st.subheader("📚 Understanding Your Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 **What the Numbers Mean**
        **Ticket Format**: Shows 6 main numbers + bonus + powerball
        **Sum**: Total of your 6 main numbers (good range: 70-180)
        **Odd/Even**: Balance between odd and even numbers
        **Last Digits**: Ending numbers of your main numbers

        ### 💰 **Expected Value**
        This is **NOT** your actual winnings! It's a mathematical estimate of:
        - How much you'd win/lose if you played millions of times
        - Always negative for lotteries (house edge)
        - Shown for educational purposes only
        """)

    with col2:
        st.markdown("""
        ### 🎲 **Why Different Tickets?**
        Each ticket is **diverse** to maximize your coverage:
        - Different number combinations
        - Various sum totals
        - Mix of odd/even patterns
        - Spread across number ranges

        ### 🎮 **Playing Strategy**
        - **Multiple tickets** = multiple chances
        - **Different patterns** = different ways to win
        - **Cost vs Coverage** = your choice
        - **Have fun!** = most important
        """)

    # Transparency section
    with st.expander("🔍 Technical Details (Advanced)"):
        if use_tft:
            st.markdown(f"""
            **🎯 Using AI-Powered Predictions**
            - **Mode:** {tft_mode.replace('_', ' ').title()}
            - **Models:** Three separate BiLSTM models (Main/Bonus/Powerball)
            - **Training:** Based on {len(optimizer.data)} historical draws
            - **Features:** Bidirectional LSTM with attention mechanism
            """)
        else:
            st.markdown("""
            **🎲 Using Uniform Random Sampling**
            - **Method:** Cryptographically seeded uniform distribution
            - **Fairness:** All numbers have equal theoretical probability
            - **Optimization:** Diversity constraints and pattern avoidance
            """)

        st.markdown("""
        **📈 Expected Value Analysis**
        - Accounts for co-winner scenarios using Poisson approximation
        - Based on official NZ Powerball prize structure
        - All figures assume $1.50 per line (Lotto $0.70 + Powerball $0.80)

        **⚠️ Important Disclaimer**
        Lottery games are games of chance. This optimization provides no guarantee
        of winning and should not be considered investment advice. Play responsibly!
        """)

    # Export functionality
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        # Export tickets as CSV
        csv_data = optimizer.export_portfolio(portfolio)
        st.download_button(
            label="📊 Download Tickets (CSV)",
            data=csv_data.to_csv(index=False),
            file_name=f"nz_powerball_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Export summary as text
        summary_text = f"""
NZ Powerball Optimized Portfolio
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Run ID: {optimizer.run_id}
Seed: {optimizer.seed}
Mode: {'BiLSTM (' + tft_mode + ')' if use_tft else 'Uniform'}

Portfolio Summary:
- Total Tickets: {len(portfolio)}
- Total Investment: ${len(portfolio) * 1.50:.2f}
- Expected Value: ${expected_value_analysis['expected_value']:.2f}
- Expected Profit: ${expected_value_analysis['expected_profit']:.2f}
- ROI: {expected_value_analysis['return_on_investment']:.6f}

Tickets:
"""
        for i, ticket in enumerate(portfolio, 1):
            summary_text += f"{i}. Main: {ticket['main_balls']}, Bonus: {ticket.get('bonus', 'N/A')}, PB: {ticket['powerball']}\n"

        st.download_button(
            label="📝 Download Summary (TXT)",
            data=summary_text,
            file_name=f"nz_powerball_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ===== DISPLAY FUNCTIONS =====

def display_tickets_cards(portfolio):
    """Display tickets in the original beautiful card format."""
    # Add custom CSS for enhanced ticket display (Light Mode Optimized)
    st.markdown("""
    <style>
        .lottery-ball {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            font-weight: 700;
            font-size: 1.2rem;
            margin: 0.25rem;
            border: 2px solid;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        .lottery-ball:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }
        .main-ball {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            border-color: #ee5a24;
        }
        .bonus-ball {
            background: linear-gradient(135deg, #ffa726, #fb8c00);
            color: white;
            border-color: #fb8c00;
        }
        .powerball {
            background: linear-gradient(135deg, #ab47bc, #8e24aa);
            color: white;
            border-color: #8e24aa;
        }
        .ticket-container {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 2px solid #dee2e6;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        .ticket-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }
        .ticket-header {
            text-align: center;
            margin-bottom: 1rem;
        }
        .ticket-number {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .stats-card {
            background: #ffffff;
            border-radius: 15px;
            padding: 1rem;
            margin-top: 1rem;
            border: 2px solid #dee2e6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stat-item:hover {
            background: #e9ecef;
        }
        .stat-label {
            font-weight: 600;
            color: #495057;
        }
        .stat-value {
            font-weight: 700;
            color: #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)

    for i, ticket in enumerate(portfolio, 1):
        # Ticket container with gradient background
        st.markdown(f"""
        <div class="ticket-container">
            <div class="ticket-header">
                <div class="ticket-number">🎫 Ticket #{i}</div>
                <h3 style="color: #1f77b4; margin: 0.5rem 0;">Your Lucky Numbers</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main numbers section
        st.markdown("#### 🎯 Main Numbers (1-40)")
        main_balls_html = '<div style="text-align: center; margin: 1rem 0;">'
        for ball in sorted(ticket['main_balls']):
            main_balls_html += f'<div class="lottery-ball main-ball">{ball:02d}</div>'
        main_balls_html += '</div>'
        st.markdown(main_balls_html, unsafe_allow_html=True)

        # Bonus and Powerball in columns
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            bonus = ticket.get('bonus', 0)
            st.markdown("#### 🎁 Bonus Ball")
            bonus_html = f'<div style="text-align: center; margin: 1rem 0;"><div class="lottery-ball bonus-ball">{bonus:02d}</div></div>'
            st.markdown(bonus_html, unsafe_allow_html=True)

        with col2:
            st.markdown("#### ⚡ Powerball")
            pb_html = f'<div style="text-align: center; margin: 1rem 0;"><div class="lottery-ball powerball">{ticket["powerball"]:02d}</div></div>'
            st.markdown(pb_html, unsafe_allow_html=True)

        with col3:
            # Statistics card
            st.markdown("#### 📊 Statistics")
            st.markdown(f"""
            <div class="stats-card">
                <div class="stat-item">
                    <span class="stat-label">Sum of Numbers:</span>
                    <span class="stat-value">{ticket['sum']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Odd/Even Balance:</span>
                    <span class="stat-value">{ticket['odd_count']}/{ticket['even_count']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Number Range:</span>
                    <span class="stat-value">{min(ticket['main_balls'])} - {max(ticket['main_balls'])}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Average Number:</span>
                    <span class="stat-value">{ticket['sum'] // 6}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Add some spacing between tickets
        if i < len(portfolio):
            st.markdown("<br>", unsafe_allow_html=True)


def display_tickets_table(portfolio, official_main=None, official_bonus=None, official_powerball=None):
    """Display tickets in a compact, comparable table format."""
    st.markdown("""
    <style>
        .table-container {
            background: #ffffff;
            border-radius: 15px;
            padding: 1rem;
            margin: 1rem 0;
            border: 2px solid #dee2e6;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        }
        .small-ball {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            font-weight: 700;
            font-size: 0.9rem;
            margin: 0.1rem;
            border: 1px solid;
        }
        .small-main-ball {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            border-color: #ee5a24;
        }
        .small-main-ball-match {
            background: linear-gradient(135deg, #007bff, #0056b3) !important;
            color: white !important;
            border-color: #0056b3 !important;
            box-shadow: 0 0 10px rgba(0, 123, 255, 0.5) !important;
        }
        .small-bonus-ball {
            background: linear-gradient(135deg, #ffa726, #fb8c00);
            color: white;
            border-color: #fb8c00;
        }
        .small-bonus-ball-match {
            background: linear-gradient(135deg, #007bff, #0056b3) !important;
            color: white !important;
            border-color: #0056b3 !important;
            box-shadow: 0 0 10px rgba(0, 123, 255, 0.5) !important;
        }
        .small-powerball {
            background: linear-gradient(135deg, #ab47bc, #8e24aa);
            color: white;
            border-color: #8e24aa;
        }
        .small-powerball-match {
            background: linear-gradient(135deg, #007bff, #0056b3) !important;
            color: white !important;
            border-color: #0056b3 !important;
            box-shadow: 0 0 10px rgba(0, 123, 255, 0.5) !important;
        }
        .table-header {
            background: linear-gradient(135deg, #1f77b4, #17a2b8);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: 700;
            font-size: 1.2rem;
        }
        .table-row {
            display: flex;
            align-items: center;
            padding: 1rem;
            border-bottom: 1px solid #dee2e6;
            transition: all 0.3s ease;
        }
        .table-row:hover {
            background: #f8f9fa;
            transform: translateX(5px);
        }
        .table-cell {
            flex: 1;
            text-align: center;
            font-weight: 600;
        }
        .ticket-id {
            font-weight: 700;
            color: #1f77b4;
            font-size: 1.1rem;
        }
        .numbers-display {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.25rem;
        }
        .match-indicator {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: bold;
            margin-left: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="table-container">', unsafe_allow_html=True)
    st.markdown('<div class="table-header">📊 Your Optimized Tickets - Compact View</div>', unsafe_allow_html=True)

    # Table header
    st.markdown("""
    <div style="display: flex; align-items: center; padding: 0.75rem; background: #e9ecef; border-radius: 8px; margin-bottom: 0.5rem; font-weight: 700; color: #495057;">
        <div style="flex: 0.5; text-align: center;">#</div>
        <div style="flex: 2; text-align: center;">Main Numbers</div>
        <div style="flex: 1; text-align: center;">Bonus</div>
        <div style="flex: 1; text-align: center;">Powerball</div>
        <div style="flex: 1; text-align: center;">Sum</div>
        <div style="flex: 1; text-align: center;">O/E</div>
    </div>
    """, unsafe_allow_html=True)

    # Show comparison info if official numbers provided
    has_official = official_main and len(official_main) == 6
    if has_official:
        st.info(f"🎯 **Comparing with official draw**: {sorted(official_main)} + Bonus: {official_bonus} + PB: {official_powerball} (Matches highlighted in blue)")

    # Table rows
    for i, ticket in enumerate(portfolio, 1):
        main_numbers_html = '<div class="numbers-display">'
        matches = 0
        
        for ball in sorted(ticket['main_balls']):
            # Check if this ball matches official numbers
            is_match = has_official and ball in official_main
            ball_class = "small-main-ball-match" if is_match else "small-main-ball"
            if is_match:
                matches += 1
            main_numbers_html += f'<div class="small-ball {ball_class}">{ball:02d}</div>'
        main_numbers_html += '</div>'
        
        # Check bonus ball match
        bonus_match = has_official and ticket.get('bonus', 0) == official_bonus
        bonus_class = "small-bonus-ball-match" if bonus_match else "small-bonus-ball"
        if bonus_match:
            matches += 0.3  # Bonus ball match
        
        # Check powerball match
        pb_match = has_official and ticket['powerball'] == official_powerball
        pb_class = "small-powerball-match" if pb_match else "small-powerball"
        if pb_match:
            matches += 0.5  # Powerball match
        
        # Add match indicator
        match_indicator = ""
        if has_official:
            main_matches = int(matches - (0.3 if bonus_match else 0) - (0.5 if pb_match else 0))
            if main_matches >= 3:
                extra_text = ""
                if bonus_match and pb_match:
                    extra_text = " + Bonus + PB!"
                elif bonus_match:
                    extra_text = " + Bonus!"
                elif pb_match:
                    extra_text = " + PB!"
                match_indicator = f'<span class="match-indicator">{main_matches}{extra_text}</span>'
            elif bonus_match or pb_match:
                if bonus_match and pb_match:
                    match_indicator = '<span class="match-indicator">Bonus + PB match!</span>'
                elif bonus_match:
                    match_indicator = '<span class="match-indicator">Bonus match!</span>'
                elif pb_match:
                    match_indicator = '<span class="match-indicator">PB match!</span>'

        st.markdown(f"""
        <div class="table-row">
            <div class="table-cell ticket-id">#{i}</div>
            <div class="table-cell">{main_numbers_html}</div>
            <div class="table-cell">
                <div class="small-ball {bonus_class}">{ticket.get('bonus', 0):02d}</div>
            </div>
            <div class="table-cell">
                <div class="small-ball {pb_class}">{ticket['powerball']:02d}</div>
            </div>
            <div class="table-cell">{ticket['sum']}</div>
            <div class="table-cell">{ticket['odd_count']}/{ticket['even_count']}{match_indicator}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def display_tickets_list(portfolio):
    """Display tickets in a fast, compact list format."""
    st.markdown("""
    <style>
        .list-container {
            background: #ffffff;
            border-radius: 15px;
            padding: 1rem;
            margin: 1rem 0;
            border: 2px solid #dee2e6;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        }
        .list-header {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: 700;
            font-size: 1.2rem;
        }
        .list-item {
            display: flex;
            align-items: center;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 10px;
            border: 1px solid #dee2e6;
            transition: all 0.3s ease;
        }
        .list-item:hover {
            background: linear-gradient(135deg, #e9ecef, #dee2e6);
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .list-number {
            font-weight: 700;
            color: #1f77b4;
            font-size: 1.1rem;
            min-width: 40px;
            text-align: center;
        }
        .list-numbers {
            flex: 1;
            font-family: 'Courier New', monospace;
            font-weight: 600;
            color: #495057;
            font-size: 1rem;
        }
        .list-stats {
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
            color: #6c757d;
        }
        .stat-chip {
            background: #ffffff;
            padding: 0.25rem 0.5rem;
            border-radius: 15px;
            border: 1px solid #dee2e6;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="list-container">', unsafe_allow_html=True)
    st.markdown('<div class="list-header">📜 Your Optimized Tickets - Fast View</div>', unsafe_allow_html=True)

    for i, ticket in enumerate(portfolio, 1):
        main_nums_str = ' '.join([f'{n:02d}' for n in sorted(ticket['main_balls'])])
        bonus = ticket.get('bonus', 0)
        pb = ticket['powerball']

        st.markdown(f"""
        <div class="list-item">
            <div class="list-number">#{i}</div>
            <div class="list-numbers">
                🎯 {main_nums_str} | 🎁 {bonus:02d} | ⚡ {pb:02d}
            </div>
            <div class="list-stats">
                <span class="stat-chip">Sum: {ticket['sum']}</span>
                <span class="stat-chip">O/E: {ticket['odd_count']}/{ticket['even_count']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

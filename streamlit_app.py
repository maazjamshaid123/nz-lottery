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
    from main import NZPowerballTicketOptimizer, TFT_AVAILABLE
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import main module: {e}")
    IMPORT_SUCCESS = False
    TFT_AVAILABLE = False
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
    🎲 <strong>Smart Lottery Optimization</strong><br>
    Generate optimized lottery tickets using advanced algorithms and machine learning.
    Upload your historical data and get AI-powered predictions for better chances!
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Help section
        with st.expander("❓ Help & Instructions", expanded=False):
            st.markdown("""
            ### 🎯 **How It Works**
            1. **Upload** your NZ Powerball historical data (CSV format)
            2. **Choose** prediction method (Uniform or AI)
            3. **Configure** number of tickets and settings
            4. **Generate** optimized lottery tickets!

            ### 📊 **Prediction Methods**
            - **🎲 Uniform**: Equal chance for all numbers (like official lottery)
            - **🎯 AI**: Uses machine learning to find patterns in historical data

            ### 🤖 **AI Sampling Explained**
            When you enable AI, it can predict numbers in two ways:

            **📈 Probabilistic** (Recommended):
            - Uses weighted probabilities based on AI predictions
            - Accounts for uncertainty and historical patterns
            - More diverse ticket selection

            **🎯 Deterministic**:
            - Picks the top predicted numbers directly
            - More focused on AI's "best guesses"
            - Less diverse but potentially more targeted

            ### 💡 **Tips**
            - Start with **Uniform** if you're new to this
            - Try **AI Probabilistic** for more variety
            - More historical data = better AI predictions
            - Results are for entertainment only!
            """)

        # FAQ section
        with st.expander("❓ Frequently Asked Questions", expanded=False):
            st.markdown("""
            ### 🎲 **What is Uniform Sampling?**
            This is the **fair way** to pick numbers - just like the official lottery! Every number has an equal chance of being selected. It's completely random and unbiased.

            ### 🎯 **What does AI do?**
            AI looks at your historical lottery data and tries to find **patterns** or **trends**. For example:
            - Are certain numbers more common on Wednesdays?
            - Do some numbers appear together more often?
            - Are there seasonal patterns?

            ### 📊 **What's the difference between Probabilistic and Deterministic?**
            - **Probabilistic**: AI gives "hints" about which numbers might be better, but still includes a good mix
            - **Deterministic**: AI picks exactly what it thinks are the best numbers

            ### 💰 **What is Expected Value?**
            This is a mathematical calculation that estimates your **long-term average winnings/losses**. In lotteries, it's always negative (you lose money over time), but it's shown for educational purposes.

            ### 🎫 **How many tickets should I buy?**
            That's up to you! More tickets = more chances, but higher cost. We recommend starting with 5 tickets and seeing how you feel.

            ### 🔄 **Can I get the same results again?**
            Yes! Use the "Random Seed" in Advanced Settings to get reproducible results, like saving your game.

            ### ⚠️ **Is this guaranteed to win?**
            **NO!** This is for entertainment only. Lottery games are random, and no system can guarantee wins. Play responsibly!

            ### 📱 **What data format do I need?**
            Your CSV should have columns: 1, 2, 3, 4, 5, 6, Power Ball, Bonus Ball (optional).
            We include a sample file to show you the format.

            ### 🚀 **Getting Started Tips**
            1. **Download sample data** to try it out first
            2. **Start with Uniform** mode (easiest)
            3. **Try 3-5 tickets** to begin
            4. **Experiment with AI** once you're comfortable
            5. **Have fun!** Remember, it's about entertainment 🎉
            """)

        # File upload
        st.subheader("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose CSV file with historical draws",
            type=['csv'],
            help="Upload your NZ Powerball historical data in CSV format. Click '❓ Help & Instructions' above for details."
        )


        # Prediction mode
        st.subheader("🎯 Prediction Mode")

        if TFT_AVAILABLE:
            use_tft = st.checkbox(
                "🎯 Enable AI Predictions",
                value=False,
                help="Use artificial intelligence to analyze patterns in your historical data and make smarter predictions. This is completely optional!"
            )
        else:
            use_tft = False
            st.info("ℹ️ AI features require extra packages. Run: pip install pytorch-forecasting pytorch-lightning scikit-learn")

        if use_tft and TFT_AVAILABLE:
            st.markdown("##### 🤖 AI Prediction Style")
            tft_mode = st.selectbox(
                "Choose AI approach:",
                ["probabilistic", "deterministic"],
                index=0,
                help="""
                🎲 Probabilistic: Uses AI predictions as 'hints' for more variety (Recommended for beginners)
                🎯 Deterministic: Follows AI predictions more closely for focused picks
                """,
                format_func=lambda x: "🎲 More Variety (Probabilistic)" if x == "probabilistic" else "🎯 More Focused (Deterministic)"
            )

            # Add explanation based on selection
            if tft_mode == "probabilistic":
                st.info("📈 **Probabilistic**: AI gives 'hints' about which numbers might be better, but still includes a mix for safety.")
            else:
                st.info("🎯 **Deterministic**: AI picks the numbers it thinks are most likely - more aggressive approach.")
        else:
            tft_mode = "probabilistic"

        # Ticket configuration
        st.subheader("🎫 Ticket Settings")
        num_tickets = st.slider(
            "Number of Tickets to Generate",
            min_value=1,
            max_value=20,
            value=5,
            help="How many different lottery tickets you want to create. More tickets = more chances, but higher cost!"
        )

        # Show cost estimate
        ticket_cost = num_tickets * 1.50
        st.info(f"💰 **Estimated Cost**: ${ticket_cost:.2f} NZD (${1.50} per ticket)")

        # Advanced settings
        with st.expander("🔧 Advanced Settings (Optional)"):
            st.markdown("**These are technical settings - you can usually leave them as default!**")

            custom_seed = st.text_input(
                "🎲 Random Seed (for testing)",
                placeholder="Leave empty for random",
                help="Makes results reproducible for testing. Like saving your game progress!"
            )

            jackpot_amount = st.number_input(
                "💰 Current Jackpot Amount (NZD)",
                min_value=1000000,
                max_value=100000000,
                value=50000000,
                step=1000000,
                help="The current jackpot amount. Used to calculate potential winnings."
            )

            estimated_sales = st.number_input(
                "👥 Estimated Tickets Sold",
                min_value=100000,
                max_value=10000000,
                value=400000,
                step=50000,
                help="Rough estimate of how many tickets are usually sold. Affects co-winner calculations."
            )

        # Generate button
        st.markdown("---")
        generate_button = st.button(
            "🚀 Generate My Lucky Tickets!",
            type="primary",
            use_container_width=True
        )

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
            generate_tickets(df, use_tft, tft_mode, num_tickets, custom_seed, jackpot_amount, estimated_sales)

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

def generate_tickets(df, use_tft, tft_mode, num_tickets, custom_seed, jackpot_amount, estimated_sales):
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
            st.success("🤖 AI mode ON (TFT)")
        else:
            st.warning("🎲 Uniform mode (no TFT)")

        # Set TFT mode if using TFT
        if use_tft:
            optimizer.tft_mode = tft_mode

        status_text.text("🎯 Generating optimized tickets...")
        progress_bar.progress(60)

        # Generate portfolio
        portfolio = optimizer.generate_ticket_portfolio(
            num_tickets=num_tickets,
            diversity_constraints={
                'max_consecutive': 3,
                'balance_odd_even': True,
                'avoid_popular_patterns': True,
                'sum_range': (70, 180),
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
        display_results(optimizer, portfolio, expected_value_analysis, use_tft, tft_mode, "📋 Table View (Compact)")

    except Exception as e:
        st.error(f"❌ Error during optimization: {str(e)}")
        st.info("💡 Try with simpler settings or check your data format")

    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_csv_path)
        except:
            pass

def display_results(optimizer, portfolio, expected_value_analysis, use_tft, tft_mode, display_mode):
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
    display_tickets_table(portfolio)

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
            - **Models:** Three separate TFT models (Main/Bonus/Powerball)
            - **Training:** Based on {len(optimizer.data)} historical draws
            - **Features:** Time-series patterns, recency, rolling statistics
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
Mode: {'TFT (' + tft_mode + ')' if use_tft else 'Uniform'}

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


def display_tickets_table(portfolio):
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
        .small-bonus-ball {
            background: linear-gradient(135deg, #ffa726, #fb8c00);
            color: white;
            border-color: #fb8c00;
        }
        .small-powerball {
            background: linear-gradient(135deg, #ab47bc, #8e24aa);
            color: white;
            border-color: #8e24aa;
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

    # Table rows
    for i, ticket in enumerate(portfolio, 1):
        main_numbers_html = '<div class="numbers-display">'
        for ball in sorted(ticket['main_balls']):
            main_numbers_html += f'<div class="small-ball small-main-ball">{ball:02d}</div>'
        main_numbers_html += '</div>'

        st.markdown(f"""
        <div class="table-row">
            <div class="table-cell ticket-id">#{i}</div>
            <div class="table-cell">{main_numbers_html}</div>
            <div class="table-cell">
                <div class="small-ball small-bonus-ball">{ticket.get('bonus', 0):02d}</div>
            </div>
            <div class="table-cell">
                <div class="small-ball small-powerball">{ticket['powerball']:02d}</div>
            </div>
            <div class="table-cell">{ticket['sum']}</div>
            <div class="table-cell">{ticket['odd_count']}/{ticket['even_count']}</div>
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

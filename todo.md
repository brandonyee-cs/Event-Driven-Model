# TODO: Model Validation Implementation

## Current State vs Target State

### What We Have
- Basic GARCH/GJR-GARCH implementation
- Three-phase volatility model with stochastic components
- RVR calculation with simple optimistic bias
- Basic analysis for FDA and earnings events
- Simple hypothesis testing for RVR peaks

### What We Need
- Full theoretical model from the paper
- Two-risk framework with explicit separation
- Multi-period portfolio optimization
- Market equilibrium with heterogeneous investors
- Comprehensive statistical validation
- Complete reproduction of paper results

## 1. Core Theoretical Model Implementation

### 1.1 Two-Risk Framework
- [ ] Implement explicit separation of directional news risk (ε) and impact uncertainty (η)
- [ ] Create classes for modeling each risk component separately
- [ ] Implement the decomposition: Re,t = μe,t + σe(t)εt where σe(t) embodies impact uncertainty
- [ ] Add methods to measure and track each risk component over time

### 1.2 Mean-Variance Optimization Framework
- [ ] Implement the multi-period portfolio optimization problem (Equation 10)
- [ ] Create solver for optimal portfolio weights (Equations 14-15)
- [ ] Handle three cases: we,t > we,t-1, we,t < we,t-1, we,t = we,t-1
- [ ] Implement dynamic rebalancing logic across time periods t=0 to t=4

### 1.3 Heterogeneous Investors
- [ ] Create classes for three investor types:
  - Informed investors (accurate It, lower b0)
  - Uninformed investors (noisier It, higher b0)
  - Liquidity traders (non-information based trading)
- [ ] Implement different information quality parameters for each type
- [ ] Model asymmetric constraints for liquidity traders

### 1.4 Transaction Costs
- [ ] Implement asymmetric transaction costs (τb > τs)
- [ ] Add cost calculation based on position changes: τi|we,t - we,t-1|Wt
- [ ] Integrate transaction costs into portfolio optimization
- [ ] Model higher costs during high-uncertainty periods

## 2. Market Equilibrium and Price Dynamics

### 2.1 Market Clearing Mechanism
- [ ] Implement aggregate demand function (Equation 20)
- [ ] Create market clearing condition: De,t(Pe,t) = Se
- [ ] Solve for equilibrium prices iteratively
- [ ] Track price evolution across event phases

### 2.2 Expectation Formation
- [ ] Implement biased expectation formation (Equation 7-8)
- [ ] Model time-varying bias parameter bt
- [ ] Implement smooth bias transition functions (already partially done)
- [ ] Calibrate κ parameter for bias sensitivity to volatility

### 2.3 Impact Uncertainty Measurement
- [ ] Implement impact uncertainty proxy (Equation 9)
- [ ] Track conditional volatility innovations
- [ ] Create metrics for unexpected volatility changes

## 3. Enhanced Volatility Modeling

### 3.1 Unified Volatility Process
- [ ] Ensure proper integration of GJR-GARCH with three-phase adjustments
- [ ] Validate phase transition smoothness
- [ ] Calibrate k1, k2, Δt1, Δt2, Δt3 parameters from data
- [ ] Add parameter uncertainty/confidence intervals

### 3.2 Stochastic Components
- [ ] Enhance stochastic noise modeling to be more realistic
- [ ] Implement event-specific volatility patterns
- [ ] Add regime-switching capabilities for different market conditions

## 4. Empirical Validation

### 4.1 Data Enhancement
- [ ] Implement proper data cleaning for extreme outliers
- [ ] Add market microstructure variables (bid-ask spreads, order flow)
- [ ] Include control variables (market cap, sector, etc.)
- [ ] Create event-specific feature engineering

### 4.2 Statistical Testing
- [ ] Implement formal hypothesis tests with proper p-values
- [ ] Add bootstrap confidence intervals for all key metrics
- [ ] Implement Fama-MacBeth regressions for cross-sectional analysis
- [ ] Add robustness checks across different time periods
- [ ] Implement HAC standard errors for time-series tests

### 4.3 Model Calibration
- [ ] Implement maximum likelihood estimation for all model parameters
- [ ] Add Bayesian estimation options
- [ ] Create parameter stability tests
- [ ] Implement rolling window calibration

## 5. Risk-Adjusted Performance Metrics

### 5.1 Enhanced RVR Calculation
- [ ] Implement exact RVR formula from paper with all components
- [ ] Add Sharpe ratio calculations with proper annualization
- [ ] Create Sortino ratio and other downside risk measures
- [ ] Implement rolling window risk metrics

### 5.2 Phase-Specific Analysis
- [ ] Create automated phase detection algorithms
- [ ] Implement phase-conditional performance metrics
- [ ] Add statistical tests for phase differences
- [ ] Create transition probability matrices between phases

## 6. Validation and Backtesting

### 6.1 Out-of-Sample Testing
- [ ] Implement proper train/test/validation splits
- [ ] Create walk-forward analysis framework
- [ ] Add cross-validation for parameter selection
- [ ] Implement portfolio performance tracking

### 6.2 Monte Carlo Simulation
- [ ] Create simulation framework for model validation
- [ ] Generate synthetic data matching model assumptions
- [ ] Test model performance under various scenarios
- [ ] Implement stress testing for extreme events

### 6.3 Benchmark Comparisons
- [ ] Compare against simple buy-and-hold strategies
- [ ] Benchmark against standard GARCH models
- [ ] Compare with other event-driven strategies
- [ ] Calculate information ratios and alpha

## 7. Reporting and Visualization

### 7.1 Comprehensive Results
- [ ] Generate all tables from the paper (phase statistics, etc.)
- [ ] Create publication-quality figures
- [ ] Add interactive visualizations for key results
- [ ] Generate LaTeX tables for easy paper integration

### 7.2 Diagnostic Plots
- [ ] QQ plots for return distributions
- [ ] Volatility clustering visualization
- [ ] Phase transition heat maps
- [ ] 3D surface plots for parameter sensitivity

## 8. Code Quality and Infrastructure

### 8.1 Testing
- [ ] Add comprehensive unit tests for all model components
- [ ] Create integration tests for full pipeline
- [ ] Add performance benchmarks
- [ ] Implement continuous integration

### 8.2 Documentation
- [ ] Add detailed docstrings matching paper notation
- [ ] Create usage examples for each component
- [ ] Add theoretical background in code comments
- [ ] Create API documentation

### 8.3 Performance Optimization
- [ ] Profile code for bottlenecks
- [ ] Parallelize Monte Carlo simulations
- [ ] Optimize data loading and processing
- [ ] Add GPU support for intensive calculations

## 9. Extensions and Robustness

### 9.1 Additional Event Types
- [ ] Extend to M&A announcements
- [ ] Add central bank announcements
- [ ] Include product launches
- [ ] Test on international markets

### 9.2 Model Extensions
- [ ] Add jump components to volatility
- [ ] Implement time-varying correlation
- [ ] Add market microstructure noise
- [ ] Include overnight vs intraday effects

### 9.3 Alternative Specifications
- [ ] Test with different utility functions
- [ ] Implement prospect theory preferences
- [ ] Add ambiguity aversion
- [ ] Test different information structures

## 10. Reproducibility

### 10.1 Data Pipeline
- [ ] Create automated data download scripts
- [ ] Add data versioning
- [ ] Implement data quality checks
- [ ] Create synthetic data generator for testing

### 10.2 Results Reproduction
- [ ] Set random seeds for all stochastic components
- [ ] Create scripts to reproduce all paper figures
- [ ] Add checksums for key results
- [ ] Create Docker container for environment

## 11. Specific Paper Results Validation

### 11.1 Key Empirical Findings
- [ ] Validate 4.4x RVR amplification for FDA approvals
- [ ] Validate 9.5x RVR enhancement for earnings announcements
- [ ] Reproduce exact phase-specific RVR values from paper
- [ ] Verify statistical significance at p < 0.001 levels

### 11.2 Cross-Sectional Validation
- [ ] Test across market cap quintiles
- [ ] Validate across sector classifications
- [ ] Test in bull vs bear market conditions
- [ ] Verify consistency across different time periods

### 11.3 Methodological Validation
- [ ] Implement 80/20 temporal split as described
- [ ] Test parameter sensitivity with exact ranges from paper
- [ ] Validate across GARCH vs GJR-GARCH specifications
- [ ] Implement conservative significance threshold (α = 0.15)

## Priority Order

1. **High Priority (Core Model)**
   - Two-risk framework implementation
   - Mean-variance optimization
   - Market equilibrium mechanics
   - Enhanced statistical testing

2. **Medium Priority (Validation)**
   - Comprehensive backtesting
   - Monte Carlo simulations
   - Robustness checks
   - Alternative specifications

3. **Low Priority (Extensions)**
   - Additional event types
   - International markets
   - GPU optimization
   - Interactive visualizations

## Implementation Milestones

### Phase 1: Core Model (4-6 weeks)
- Implement two-risk framework
- Create investor heterogeneity classes
- Build market equilibrium solver
- Integrate with existing GARCH models

### Phase 2: Empirical Validation (3-4 weeks)
- Enhance data pipeline
- Implement all statistical tests
- Reproduce key paper results
- Generate all required visualizations

### Phase 3: Robustness & Extensions (2-3 weeks)
- Add Monte Carlo framework
- Implement backtesting suite
- Create parameter sensitivity analysis
- Add alternative model specifications

### Phase 4: Documentation & Polish (1-2 weeks)
- Complete all documentation
- Add comprehensive tests
- Create reproducibility scripts
- Prepare final deliverables

## Key Success Metrics

1. **Model Accuracy**
   - RVR amplification within 10% of paper values
   - Statistical significance matches paper (p < 0.001)
   - Phase patterns clearly visible in data

2. **Code Quality**
   - >90% test coverage
   - All results reproducible with fixed seeds
   - Performance benchmarks meet targets

3. **Documentation**
   - All functions documented with paper references
   - Clear mapping between code and paper equations
   - Complete usage examples for all components

## File Structure Changes

### New Files to Create
- `src/two_risk_framework.py` - Directional news risk and impact uncertainty
- `src/portfolio_optimization.py` - Mean-variance optimization solver
- `src/market_equilibrium.py` - Market clearing and price dynamics
- `src/investors.py` - Heterogeneous investor classes
- `src/transaction_costs.py` - Asymmetric cost modeling
- `src/statistical_tests.py` - Comprehensive testing suite
- `src/monte_carlo.py` - Simulation framework
- `src/backtesting.py` - Out-of-sample validation
- `tests/` - Unit and integration tests for all components

### Files to Modify
- `src/models.py` - Enhance GARCH models with two-risk integration
- `src/event_processor.py` - Add market equilibrium and investor types
- `testhyp1.py` - Expand to full hypothesis testing suite
- Add proper parameter calibration to all analysis scripts

## Additional Dependencies

### New Python Packages Needed
- `scipy.optimize` - For advanced optimization (already used)
- `arch` - For professional GARCH modeling
- `cvxpy` - For convex optimization problems
- `joblib` - For parallel processing
- `pytest` - For comprehensive testing
- `sphinx` - For documentation generation
- Additional statistical packages as needed

### Infrastructure Requirements
- Set up CI/CD pipeline
- Configure Docker environment
- Set up data storage solution
- Implement version control for data and results

## Data Requirements

### Primary Data Sources (as per paper)
- **I/B/E/S Actuals History** - Earnings announcement dates and details
- **FDA Approval Records** - Drug approval dates and companies
- **CRSP Daily Stock File** - Prices, returns, volumes, microstructure data
- **Control Variables** - Market cap, sector classifications, etc.

### Data Quality Requirements
- Minimum 20 observations per event window
- Winsorization at 1st and 99th percentiles
- Handle missing data appropriately
- Verify ticker symbol consistency across sources

### Sample Size Targets
- ~89,500 event-date pairs (as mentioned in paper)
- Sufficient events for cross-sectional analysis
- Adequate time span for temporal validation

## Implementation Challenges & Risks

### Technical Challenges
- Market equilibrium solver convergence issues
- GARCH estimation numerical stability
- High computational cost for Monte Carlo
- Memory management for large datasets

### Theoretical Challenges
- Parameter identification in two-risk framework
- Calibrating investor heterogeneity parameters
- Ensuring model consistency across phases
- Avoiding overfitting to specific event types

### Validation Challenges
- Reproducing exact paper results
- Handling data discrepancies
- Ensuring statistical power
- Managing multiple testing issues

### Mitigation Strategies
- Start with simplified versions and build complexity
- Use robust optimization techniques
- Implement extensive logging and debugging
- Create fallback options for each component
- Regular validation against paper benchmarks
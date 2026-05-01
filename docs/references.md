# References

## Direct indexing and tax-loss harvesting

- Aperio Group (BlackRock). "Personalized Indexing." White papers available
  at the Aperio research portal. Aperio is the industry pioneer for
  customized index-tracking portfolios with tax-management overlays.

- Vanguard Personalized Indexing Management. "Personalized Indexing: A
  Portfolio Construction Plan." (2022). Khang, Cummings, Paradise, O'Connor.

- Parametric Portfolio Associates. Numerous research papers on after-tax
  optimization and tax-loss harvesting alpha generation.

- Elm Wealth. "Robbing Peter to Pay Paul: A(nother) Look at Long/Short
  Direct Index Tax-Loss Harvesting." A skeptical examination of LSDI
  programs that includes a useful simulation model in the appendix.

- Internal Revenue Code Section 1091 (the wash-sale rule).

## Factor models

- Fama, E. F. and French, K. R. (1993). "Common risk factors in the returns
  on stocks and bonds." Journal of Financial Economics, 33, 3-56.

- Fama, E. F. and French, K. R. (2015). "A five-factor asset pricing model."
  Journal of Financial Economics, 116, 1-22.

- Carhart, M. M. (1997). "On persistence in mutual fund performance."
  Journal of Finance, 52, 57-82. Source for the momentum factor.

- Kenneth R. French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

## Portfolio optimization

- Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P., and
  Speth, J. (2017). "Multi-Period Trading via Convex Optimization."
  Foundations and Trends in Optimization, 3(1), 1-76. The methodological
  reference for cvxportfolio.

- Black, F. and Litterman, R. (1992). "Global portfolio optimization."
  Financial Analysts Journal, 48(5), 28-43. The original Black-Litterman.

- Diamond, S., Boyd, S. (2016). "CVXPY: A Python-Embedded Modeling Language
  for Convex Optimization." Journal of Machine Learning Research, 17(83),
  1-5.

## Transaction cost modeling

- Almgren, R., Thum, C., Hauptmann, E., and Li, H. (2005). "Direct
  estimation of equity market impact." Risk, 18(7), 58-62. The
  square-root market impact model.

- Kissell, R. (2014). "The Science of Algorithmic Trading and Portfolio
  Management." Academic Press.

## Adjacent open-source projects

- cvxgrp/cvxportfolio: Stanford's portfolio optimization and backtesting
  library. The most rigorous reference implementation in the open-source
  ecosystem.

- dcajasn/Riskfolio-Lib: Comprehensive portfolio optimization library
  built on cvxpy.

- bbreslauer/wash-sale-tracker: Cost basis adjustment script for personal
  tax filing.

- redstreet/fava_tax_loss_harvester: Beancount/Fava plugin for personal
  tax-loss harvesting.

These projects served as reference points for the design of this toolkit.
None combines factor modeling, optimization, backtesting, and integrated
tax-loss harvesting in a single cohesive framework.

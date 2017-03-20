# Zmienną losowaną z proposala będzie wektor średnich m. Będziemy losować m_k z rozkładu beta ze średnią s = sum(x_n*y_nk)/sum(y_nk) o parametrach alpha = s/min(s,1-s), beta = (1-s)/min(s, 1-s). Taka normalizacja zapewni nam, że rozkład nie będzie zbiegał w szalonym tempie do brzegów. Na końcu.
from proposal import Q
# Można wyliczyć, że warunkowy rozkład w_k | m, X, Y jest postaci Dir(1+sum(y_n,1),..., 1+sum(y_n,k))
# Analogicznie Y_n|w,m,X ~ Multinomial(w * g_m(X_n)/sum_k(w_k*g_m(X_n)))
from distributions import rW, rY

from math import floor, ceil
from pandas import DataFrame

EPS = .00001


def _releq(a, b):
    assert abs(a - b) < EPS


def _releql(l1, l2):
    assert len(l1) == len(l2)
    for a, b in zip(l1, l2):
        _releq(a, b)


def hypergeometric(k, m, n):
    return k * m/n * (1 - m/n) * (n-k) / (n-1)


def count_mi_ni(feature_column, stratum_column):
    if not min(stratum_column) == 0:
        raise ValueError(f"The minimum stratum entry must be 0, not {min(stratum_column)}.")
    ell = max(stratum_column) + 1
    if not ell >= 1:
        raise ValueError(f"There must be at least one stratum, but the maximum stratum number found is {ell - 1}.")
    if not len(set(stratum_column)) == ell:
        raise ValueError(f"Strata must should be numbered 0, …, l-1, and all strata should be non-empty. However, "
                         f"a there are the following strata numbers: {set(stratum_column)}.")
    
    ni = [0] * ell
    mi = [0] * ell
    for stratum, feature in zip(stratum_column, feature_column):
        assert feature == 0 or feature == 1
        ni[stratum] += 1
        if feature == 1:
            mi[stratum] += 1
            
    return mi, ni


def get_layout(ni, k):
    """Given strata sizes `n_i` and panel size `k`, compute the `rho_i` and `g_i` for block rounding."""
    assert k > 0
    ell = len(ni)
    n = sum(ni)
    gi = [0] * ell
    rhoi = [0] * ell
    count = 0
    for stratum in range(ell):
        if ni[stratum] * k < n:
            raise ValueError(f"All n_i must be at least n/k, but n_{stratum} = {ni[stratum]} < {n / k} = n/k.")
        gi[stratum] = (count + ni[stratum]) * k // n - (count * k + n - 1) // n
        _releq(gi[stratum], floor((count + ni[stratum]) / (n/k)) - ceil(count / (n/k)))
        assert gi[stratum] >= 0
        rhoi[stratum] = (((count + ni[stratum]) * k) % n) / n
        _releq(rhoi[stratum], (count + ni[stratum]) / (n/k) - floor((count + ni[stratum]) / (n/k)))
        assert 0 <= rhoi[stratum] < 1
        assert rhoi[stratum] == 0 or rhoi[stratum] > EPS
        count += ni[stratum]
    assert rhoi[ell - 1] == 0
    assert sum(gi) + sum(1 for rho in rhoi if rho != 0) == k
    
    return rhoi, gi


def expvar(ni, mi, rhoi, gi):
    """Compute the expected value (over rounding) of the variance (over selection inside of strata) of the
    representation of the hidden feature.
    """
    ell = len(ni)
    if ell != len(mi) or ell != len(rhoi) or ell != len(gi):
        raise ValueError("The lengths of arguments ni, mi, rhoi, and gi must coincide.")
    
    ev = 0
    for stratum in range(ell):
        if stratum == ell - 1:
            rho = 0
        else:
            rho = rhoi[stratum]
        if stratum == 0 or rhoi[stratum - 1] == 0:
            lam = 0
        else:
            lam = 1 - rhoi[stratum - 1]
        gx = gi[stratum]
        mx = mi[stratum]
        nx = ni[stratum]

        ev += (1 - lam) * (1 - rho) * hypergeometric(gx, mx, nx)
        ev += ((1 - lam) * rho + lam * (1 - rho)) * hypergeometric(gx + 1, mx, nx)
        ev += lam * rho * hypergeometric(gx + 2, mx, nx)
    return ev


def varexp(ni, mi, rhoi):
    """Compute the variance (over rounding) of the expected value (over selection inside of strata) of the
    representation of the hidden feature.
    """
    ell = len(ni)
    ve = 0
    for stratum in range(ell - 1):
        ve += (mi[stratum]/ni[stratum] - mi[stratum+1]/ni[stratum+1])**2 * rhoi[stratum] * (1 - rhoi[stratum])
    return ve


def compute_variance(data, hidden_feature, strata, k, partial=None):
    """Compute the variance in representation of a feature, given a stratification.

    Args:
       data (DataFrame): Table containing an index column and a column with the value of `hidden_feature` as its name.
       hidden_feature (string): The column name to use as the hidden feature. `data[hidden_feature]` should be a binary
                                column with only entries 0 and 1.
       strata (DataFrame): Table containing an index column referring to the same agents as `data` and a column
                           "stratum" describing the stratification. Entries should be between 0 and (number of
                           strata - 1), each stratum is expected to be non-empty.
       partial: if "varexp" or "expvar", return only corresponding summand of variance
    """
    assert len(data) == len(strata)
    combination = data.join(strata, how="inner")
    assert len(combination) == len(data)
    
    assert len(strata) != 0
    mi, ni = count_mi_ni(combination[hidden_feature], combination["stratum"])
    rhoi, gi = get_layout(ni, k)
    
    if partial == "varexp":
        return varexp(ni, mi, rhoi)
    elif partial == "expvar":
        return expvar(ni, mi, rhoi, gi)
    else:
        return expvar(ni, mi, rhoi, gi) + varexp(ni, mi, rhoi)


def test_var():
    """Run usage examples and verify that variance computation functions work as expected."""
    rhoi, gi = get_layout([2, 2], 3)
    _releql(rhoi, [0.5, 0])
    _releql(gi, [1, 1])
    rhoi, gi = get_layout([4, 5, 6], 5)
    _releql(rhoi, [1/3, 0, 0])
    assert rhoi[1] == 0
    assert rhoi[2] == 0
    _releql(gi, [1, 1, 2])
    
    _releq(expvar([2, 2], [1, 1], [0, 0], [1, 1]), 0.5)
    _releq(expvar([3, 6, 3], [0, 3, 3], [.5, .5, 0], [1, 2, 1]), 17/40)
    
    _releq(varexp([3, 6, 9, 6], [2, 4, 6, 4], [.5, .5, 0, 0]), 0)
    
    _releq(compute_variance(DataFrame({"m": [1, 1]}), "m", DataFrame({"stratum": [0, 0]}), 1), 0)
    _releq(compute_variance(DataFrame({"m": [0, 0, 1, 1]}), "m", DataFrame({"stratum": [0, 0, 1, 1]}), 2), 0)
    _releq(compute_variance(DataFrame({"m": [0, 0, 1, 1]}), "m", DataFrame({"stratum": [0, 1, 0, 1]}), 2), 0.5)


def _stratify_successively(data, features, min_size, prestratification=None):
    if prestratification is None:
        prestratification = [list(data.index)]
    if len(features) == 0:
        return prestratification
    
    f = features[0]
    new_stratification = []
    for stratum in prestratification:
        if len(stratum) < 2 * min_size:
            new_stratification.append(stratum)
        else:
            mini = int(data[f].min())
            maxi = int(data[f].max())
            lower = mini - 1
            remaining = len(stratum)
            for upper in range(mini, maxi):
                new_index = [i for i in stratum if lower < data.loc[i, f] <= upper]
                if len(new_index) >= min_size and remaining - len(new_index) >= min_size:
                    new_stratification.append(new_index)
                    lower = upper
                    remaining -= len(new_index)
            new_index = [i for i in stratum if not data.loc[i, f] <= lower]
            assert len(new_index) == remaining
            new_stratification.append(new_index)
            
    return _stratify_successively(data, features[1:], min_size, new_stratification)


def successive_stratification(data, features, min_size):
    """Stratify by successively refining based on a sequence of features, up to a minimum size.

    Args:
        data (DataFrame): Table with index column and columns named like the elements of `features`.
        features (list of string): Column names. Stratify the agents by the entry in data[features[0]], then subdivide
                                   based on data[features[1]] and so forth, until doing so would create strata with size
                                   less than `min_size`.
        min_size (int): Minimum size for strata.

    Returns:
        DataFrame
        Table with index column matching `data` and column "stratum" with stratum number.
    """
    strat = _stratify_successively(data, features, min_size)
    index = data.index
    
    positions = {}
    for pos, i in enumerate(index):
        positions[i] = pos

    stratum = [None] * len(data)
    for stratum_name, elements in enumerate(strat):
        for x in elements:
            stratum[positions[x]] = stratum_name
    assert all(x is not None for x in stratum)

    return DataFrame({"stratum": stratum}, index=index)


def test_successive_stratification():
    d = DataFrame({"a": [0, 0, 0, 1, 1, 1, 2, 2, 2], "b": [0, 1, 2, 0, 1, 2, 0, 1, 2]})
    assert _stratify_successively(d, [], 3) == [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    assert _stratify_successively(d, ["a"], 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert _stratify_successively(d, ["a"], 1) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert _stratify_successively(d, ["b"], 3) == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    assert _stratify_successively(d, ["a"], 4) == [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    assert _stratify_successively(d, ["a","b"], 1) == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    d = DataFrame({"a": [0, 0, 0, 1, 1, 1, 2, 2, 2], "b": [0, 1, 2, 0, 1, 2, 0, 1, 2]}, index=[7, 8, 9, 10, 11, 12, 13, 15, 16])
    assert _stratify_successively(d, [], 3) == [[7, 8, 9, 10, 11, 12, 13, 15, 16]]
    d = DataFrame({"a": [0, 0, 0, 0, 0, 1]})
    assert _stratify_successively(d, ["a"], 2) == [[0, 1, 2, 3, 4, 5]]
    d = DataFrame({"a": [0, 0, 0, 0, 1, 2], "b": [1, 2, 1, 2, 1, 2]})
    assert _stratify_successively(d, ["a"], 2) == [[0, 1, 2, 3], [4, 5]]
    assert _stratify_successively(d, ["a", "b"], 2) == [[0, 2], [1, 3], [4, 5]]


def equivalent_panel_size(variance, n, m, k):
    """Compute k' such that Var(U_M^{n,k'} / k') = Var(X_M^{n,k} / k).
    
    To be exact, the resulting k' is a fractional interpolation of the variance
    formula k * m/n * (1 - m/n) * (n-k)/(n-1).
    Argument `variance` is Var(X_M^{n,k}), not normalized.
    """
    assert m > 0
    assert m < n
    
    return (k**2 * m * n * (n-m)) / (variance * n**2 * (n-1) + k**2 * m * (n-m))
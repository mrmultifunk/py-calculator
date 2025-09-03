import streamlit as st
import sympy as sp
import numpy as np
import math
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    rationalize,
)
from sympy import Matrix, Eq
from typing import Dict, List, Tuple, Optional

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Algebra & Equation Solver",
    page_icon="🧮",
    layout="wide",
)

# ------------------------------
# Styling
# ------------------------------
st.markdown(
    """
    <style>
      .result-card {border:1px solid rgba(0,0,0,0.1); padding:1rem; border-radius:12px; background: rgba(250,250,250,0.7);}
      .hint {opacity:0.8; font-size:0.9rem}
      .example-badge {display:inline-block; margin:0 .25rem .25rem 0; padding:.25rem .5rem; border:1px solid #ddd; border-radius:999px; cursor:pointer;}
      .footer-note {font-size:0.9rem; opacity:0.8}
      .section-title {margin-top: .25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Allowed names for safe parsing
# ------------------------------
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    rationalize,
)

# Core constants and functions
ALLOWED_CONSTS = {
    "E": sp.E,
    "pi": sp.pi,
    "Pi": sp.pi,
    "I": sp.I,
    "oo": sp.oo,
    "nan": sp.nan,
}

ALLOWED_FUNCS = {
    # elementary
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "sign": sp.sign,
    # integer & combinatorics
    "factorial": sp.factorial, "gamma": sp.gamma, "binomial": sp.binomial,
    # rounding
    "floor": sp.floor, "ceiling": sp.ceiling,
    # optional numeric rounding (non-symbolic)
    "round": round,
    # misc
    "re": sp.re, "im": sp.im, "arg": sp.arg,
}

# Pre-create a pool of symbols that users commonly type
COMMON_SYMBOL_NAMES = [
    # lowercase letters
    *[chr(c) for c in range(ord('a'), ord('z')+1)],
    # some typical multi-letter names
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "lambda",
    "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
    # time and common math vars
    "t", "u", "v", "w",
]
BASE_SYMBOLS: Dict[str, sp.Symbol] = {name: sp.Symbol(name) for name in COMMON_SYMBOL_NAMES}


def make_symbol(name: str, real: bool = None, positive: bool = None, integer: bool = None) -> sp.Symbol:
    assumptions = {}
    if real is not None:
        assumptions["real"] = real
    if positive is not None:
        assumptions["positive"] = positive
    if integer is not None:
        assumptions["integer"] = integer
    return sp.Symbol(name, **assumptions)


def build_symbol_table(user_vars: List[str], real: Optional[bool], positive: Optional[bool], integer: Optional[bool]) -> Dict[str, sp.Symbol]:
    table = dict(BASE_SYMBOLS)
    for v in user_vars:
        v = v.strip()
        if not v:
            continue
        table[v] = make_symbol(v, real=real, positive=positive, integer=integer)
    return table


def safe_parse(expr: str, symbols: Dict[str, sp.Symbol]) -> sp.Expr:
    local_dict = {**ALLOWED_CONSTS, **ALLOWED_FUNCS, **symbols}
    return parse_expr(expr, local_dict=local_dict, transformations=TRANSFORMATIONS, evaluate=False)


def parse_equation(text: str, symbols: Dict[str, sp.Symbol]) -> sp.Eq:
    """Parse 'lhs = rhs' or single expression 'expr' (treated as expr = 0)."""
    text = text.strip()
    if "==" in text:
        lhs, rhs = text.split("==", 1)
        return sp.Eq(safe_parse(lhs, symbols), safe_parse(rhs, symbols))
    if "=" in text:
        lhs, rhs = text.split("=", 1)
        return sp.Eq(safe_parse(lhs, symbols), safe_parse(rhs, symbols))
    # treat as expr = 0
    return sp.Eq(safe_parse(text, symbols), 0)


def render(obj) -> None:
    try:
        st.latex(sp.latex(obj))
    except Exception:
        st.write(obj)


def render_solutions(sol):
    if isinstance(sol, (set, sp.FiniteSet)):
        if len(sol) == 0:
            st.info("No solutions found in the selected domain.")
            return
        for s in sol:
            render(s)
    elif isinstance(sol, sp.ConditionSet):
        st.warning("Could not find a closed-form solution. Showing a condition set:")
        render(sol)
    else:
        render(sol)


# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("🧮 Algebra & Equation Solver")
st.sidebar.markdown(
    """
**Syntax tips**
- Use `^` for powers: `x^2 + 3x - 4`.
- Functions: `sin(x)`, `cos(x)`, `tan(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, `Abs(x)`, ...
- Constants: `pi`, `E`, `I` (imaginary unit).
- Equations: `lhs = rhs` or just type an expression to solve `expr = 0`.
- Systems: one equation per line.
    """
)

with st.sidebar.expander("Examples", expanded=False):
    st.code(
        """
# Equation solver
x^2 - 5x + 6 = 0

# System solver
x + y = 5
x - y = 1

# Expression tools
(x^2 - 1)/(x-1)

# Calculus
sin(x)^2

# Plotter
sin(x)/x
        """,
        language="python",
    )

st.sidebar.markdown(
    """
**Note**: This app uses [SymPy](https://www.sympy.org/) under the hood. While it handles a *lot*, some problems may require numeric methods or may not have a closed form.
    """
)

# ------------------------------
# Main UI
# ------------------------------
st.title("Algebra & Equation Solver")
st.caption("Symbolic & numeric math with SymPy — equations, systems, simplification, calculus, linear algebra, and plotting.")

# Variable & assumption controls
with st.expander("Variables & assumptions", expanded=False):
    cols = st.columns([2,1,1,1])
    with cols[0]:
        vars_text = st.text_input("Variables (comma-separated)", value="x", key="vars_text")
    with cols[1]:
        assume_real = st.selectbox("Real?", ("unspecified", "yes", "no"), index=0)
    with cols[2]:
        assume_positive = st.selectbox("Positive?", ("unspecified", "yes", "no"), index=0)
    with cols[3]:
        assume_integer = st.selectbox("Integer?", ("unspecified", "yes", "no"), index=0)

    def as_opt(v):
        return None if v == "unspecified" else (v == "yes")

    var_names = [v.strip() for v in vars_text.split(",") if v.strip()]
    SYMBOL_TABLE = build_symbol_table(
        var_names, real=as_opt(assume_real), positive=as_opt(assume_positive), integer=as_opt(assume_integer)
    )

# Tabs
TAB_EQ, TAB_SYS, TAB_EXPR, TAB_CALC, TAB_LA, TAB_PLOT = st.tabs([
    "Equation Solver", "System Solver", "Expression Tools", "Calculus", "Linear Algebra", "Plotter"
])

# ------------------------------
# Equation Solver
# ------------------------------
with TAB_EQ:
    st.subheader("Solve a single equation")
    eq_text = st.text_area("Enter an equation or expression (treated as = 0)", value="x^2 - 5x + 6 = 0", height=90)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        solve_var = st.text_input("Solve for variable", value=var_names[0] if var_names else "x", key="solve_var")
    with col2:
        domain_choice = st.selectbox("Domain", ("Reals", "Complexes"), index=0)
    with col3:
        numeric_eval = st.checkbox("Also show numeric approximation", value=True)

    domain = sp.S.Reals if domain_choice == "Reals" else sp.S.Complexes

    if st.button("Solve equation", type="primary"):
        try:
            eq = parse_equation(eq_text, SYMBOL_TABLE)
            var = SYMBOL_TABLE.get(solve_var, sp.Symbol(solve_var))
            sol = sp.solveset(eq, var, domain=domain)
            st.markdown("**Solution set:**")
            render_solutions(sol)
            if numeric_eval:
                try:
                    if isinstance(sol, (sp.FiniteSet, set)):
                        st.markdown("**Numeric:**")
                        for s in sol:
                            render(sp.N(s))
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Error while solving: {e}")

# ------------------------------
# System Solver
# ------------------------------
with TAB_SYS:
    st.subheader("Solve a system of equations")
    sys_text = st.text_area("Enter one equation per line", value="x + y = 5\nx - y = 1", height=140)
    sys_vars_text = st.text_input("Variables (order matters for numeric methods)", value=",".join(var_names) or "x,y", key="sys_vars") or "x,y")
    method = st.selectbox("Method", ("auto", "symbolic", "linear (linsolve)", "numeric (nsolve)"), index=0)

    if st.button("Solve system", key="solve_system", type="primary"):
        try:
            lines = [ln.strip() for ln in sys_text.splitlines() if ln.strip()]
            eqs = [parse_equation(ln, SYMBOL_TABLE) for ln in lines]
            vars_list = [SYMBOL_TABLE.get(v.strip(), sp.Symbol(v.strip())) for v in sys_vars_text.split(',') if v.strip()]

            sol_out = None

            if method in ("auto", "symbolic"):
                try:
                    sol_out = sp.solve(eqs, vars_list, dict=True)
                except Exception:
                    sol_out = None

            if sol_out in (None, []) and method in ("auto", "linear (linsolve)"):
                try:
                    # convert to linear system if possible
                    A, b = sp.linear_eq_to_matrix([Eq(e.lhs, e.rhs) for e in eqs], vars_list)
                    sol_lin = sp.linsolve((A, b), *vars_list)
                    sol_out = [dict(zip(vars_list, tup)) for tup in list(sol_lin)]
                except Exception:
                    pass

            if (sol_out in (None, []) or method == "numeric (nsolve)") and len(vars_list) > 0:
                # Try numeric nsolve with a default initial guess of zeros
                try:
                    guesses = [0]*len(vars_list)
                    F = [e.lhs - e.rhs for e in eqs]
                    ns = sp.nsolve(F, vars_list, guesses)
                    if not isinstance(ns, (list, tuple)):
                        ns = [ns]
                    sol_out = [dict(zip(vars_list, ns))]
                except Exception as ne:
                    if method == "numeric (nsolve)":
                        st.error(f"nsolve failed: {ne}")

            if sol_out in (None, []):
                st.info("No solution found or could not solve symbolically.")
            else:
                st.markdown("**Solutions:**")
                for i, sol in enumerate(sol_out, 1):
                    with st.container():
                        st.markdown(f"Solution {i}:")
                        for var, val in sol.items():
                            st.latex(f"{sp.latex(var)} = {sp.latex(val)}")
                        st.markdown("---")
        except Exception as e:
            st.error(f"Error while solving system: {e}")

# ------------------------------
# Expression Tools
# ------------------------------
with TAB_EXPR:
    st.subheader("Work with expressions")
    expr_text = st.text_input("Expression", value="(x^2 - 1)/(x-1)", key="expr_text")/(x-1)")
    ops = st.multiselect(
        "Operations",
        ["simplify", "factor", "expand", "cancel", "apart (partial fractions)", "trigsimp", "powsimp", "collect"],
        default=["simplify"],
    )
    collect_var = st.text_input("Collect by variable (for collect)", value="x", key="collect_var")
    subs_text = st.text_input("Substitutions (comma-separated, e.g., x=2, y=pi/3)", value="", key="subs_text")

    if st.button("Apply", key="expr_apply", type="primary"):
        try:
            e = safe_parse(expr_text, SYMBOL_TABLE)
            out = e
            if "simplify" in ops:
                out = sp.simplify(out)
            if "factor" in ops:
                out = sp.factor(out)
            if "expand" in ops:
                out = sp.expand(out)
            if "cancel" in ops:
                out = sp.cancel(out)
            if "apart (partial fractions)" in ops:
                out = sp.apart(out, SYMBOL_TABLE.get(collect_var, sp.Symbol(collect_var)))
            if "trigsimp" in ops:
                out = sp.trigsimp(out)
            if "powsimp" in ops:
                out = sp.powsimp(out)
            if "collect" in ops:
                out = sp.collect(out, SYMBOL_TABLE.get(collect_var, sp.Symbol(collect_var)))

            if subs_text.strip():
                pairs = [p.strip() for p in subs_text.split(',') if p.strip()]
                subs = {}
                for p in pairs:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        subs[safe_parse(k, SYMBOL_TABLE)] = safe_parse(v, SYMBOL_TABLE)
                out = out.subs(subs)

            st.markdown("**Result:**")
            render(out)
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------------------
# Calculus
# ------------------------------
with TAB_CALC:
    st.subheader("Derivatives, integrals, limits")
    calc_expr = st.text_input("Expression", value="sin(x)^2", key="calc_expr")^2")
    calc_var = st.text_input("Variable", value=var_names[0] if var_names else "x", key="calc_var")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        deriv_order = st.number_input("Derivative order (0 for none)", min_value=0, value=1)
    with c2:
        do_integral = st.checkbox("Indefinite integral")
    with c3:
        do_definite = st.checkbox("Definite integral (set bounds)")

    b1, b2, b3 = st.columns([1,1,1])
    with b1:
        limit_dir = st.selectbox("Limit direction", ("two-sided", "+", "-"), index=0)
    with b2:
        limit_point = st.text_input("Limit point (optional)", value="", key="limit_point")
    with b3:
        compute_limit = st.checkbox("Compute limit")

    if st.button("Compute", key="calc_compute", type="primary"):
        try:
            e = safe_parse(calc_expr, SYMBOL_TABLE)
            v = SYMBOL_TABLE.get(calc_var, sp.Symbol(calc_var))

            if deriv_order and deriv_order > 0:
                st.markdown("**Derivative:**")
                render(sp.diff(e, v, int(deriv_order)))

            if do_integral:
                st.markdown("**Indefinite integral:**")
                render(sp.integrate(e, v))

            if do_definite:
                lower = st.text_input("Lower bound", value="0", key="lower")
                upper = st.text_input("Upper bound", value="pi", key="upper")
                try:
                    a = safe_parse(lower, SYMBOL_TABLE)
                    b = safe_parse(upper, SYMBOL_TABLE)
                    st.markdown("**Definite integral:**")
                    render(sp.integrate(e, (v, a, b)))
                except Exception as be:
                    st.error(f"Invalid bounds: {be}")

            if compute_limit and limit_point.strip():
                pt = safe_parse(limit_point, SYMBOL_TABLE)
                dir_map = {"two-sided": None, "+": "+", "-": "-"}
                st.markdown("**Limit:**")
                render(sp.limit(e, v, pt, dir=dir_map[limit_dir]))

        except Exception as e:
            st.error(f"Error in calculus tab: {e}")

# ------------------------------
# Linear Algebra
# ------------------------------
with TAB_LA:
    st.subheader("Matrix & vector operations")
    st.markdown(
        "Enter a matrix like `1 2 3; 4 5 6; 7 8 9` or with newlines. You can also use expressions (e.g., `x 1; 2 x`)."
    )
    mat_text = st.text_area("Matrix", value="1 2; 3 4", height=120)

    la_cols = st.columns([1,1,1,1])
    with la_cols[0]:
        show_det = st.checkbox("determinant", value=True)
    with la_cols[1]:
        show_rank = st.checkbox("rank", value=True)
    with la_cols[2]:
        show_inverse = st.checkbox("inverse")
    with la_cols[3]:
        show_rref = st.checkbox("RREF")

    la_cols2 = st.columns([1,1,1])
    with la_cols2[0]:
        show_eigs = st.checkbox("eigenvalues")
    with la_cols2[1]:
        show_eigv = st.checkbox("eigenvectors")
    with la_cols2[2]:
        show_decomp = st.checkbox("LU/QR")

    if st.button("Compute", key="la_compute", type="primary"):
        try:
            # Parse matrix
            rows = [r.strip() for r in mat_text.replace("\n", ";").split(";") if r.strip()]
            parsed_rows = []
            for r in rows:
                entries = [e for e in r.replace(",", " ").split() if e]
                parsed_rows.append([safe_parse(e, SYMBOL_TABLE) for e in entries])
            M = Matrix(parsed_rows)

            st.markdown("**Matrix:**")
            render(M)

            if show_det:
                st.markdown("**det(M):**")
                render(M.det())
            if show_rank:
                st.markdown("**rank(M):**")
                render(M.rank())
            if show_inverse:
                if M.det() != 0:
                    st.markdown("**M^{-1}:**")
                    render(M.inv())
                else:
                    st.info("Matrix not invertible (determinant = 0)")
            if show_rref:
                st.markdown("**RREF:**")
                R, pivots = M.rref()
                render(R)
                st.markdown(f"Pivots: {pivots}")
            if show_eigs:
                st.markdown("**Eigenvalues:**")
                vals = M.eigenvals()
                for lam, mult in vals.items():
                    st.latex(fr"\lambda = {sp.latex(lam)} \;\; (mult={mult})")
            if show_eigv:
                st.markdown("**Eigenvectors:**")
                vecs = M.eigenvects()
                for lam, mult, vlist in vecs:
                    st.latex(fr"\lambda = {sp.latex(lam)} \;\; (mult={mult})")
                    for v in vlist:
                        render(v)
            if show_decomp:
                try:
                    L, U = M.LUdecomposition()[:2]
                    st.markdown("**LU decomposition:**")
                    st.markdown("L:")
                    render(L)
                    st.markdown("U:")
                    render(U)
                except Exception:
                    st.info("LU decomposition not available for this matrix.")
                try:
                    Q, R = M.QRdecomposition()
                    st.markdown("**QR decomposition:**")
                    st.markdown("Q:")
                    render(Q)
                    st.markdown("R:")
                    render(R)
                except Exception:
                    st.info("QR decomposition not available for this matrix.")
        except Exception as e:
            st.error(f"Matrix error: {e}")

# ------------------------------
# Plotter
# ------------------------------
with TAB_PLOT:
    st.subheader("Plot an expression of one variable")
    plot_expr = st.text_input("Expression", value="sin(x)/x", key="plot_expr")/x")
    plot_var = st.text_input("Variable", value=var_names[0] if var_names else "x", key="plot_var")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_min = st.number_input("x min", value=-10.0)
    with c2:
        x_max = st.number_input("x max", value=10.0)
    with c3:
        num_points = st.slider("Points", min_value=100, max_value=2000, value=600, step=100)

    if st.button("Plot", key="plot_go", type="primary"):
        try:
            e = safe_parse(plot_expr, SYMBOL_TABLE)
            v = SYMBOL_TABLE.get(plot_var, sp.Symbol(plot_var))
            f = sp.lambdify(v, e, modules=[{"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "Abs": np.abs}, "numpy"])
            xs = np.linspace(float(x_min), float(x_max), int(num_points))
            ys = f(xs)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(xs, ys)
            ax.set_xlabel(str(v))
            ax.set_ylabel("f(" + str(v) + ")")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Plotting error: {e}")

# ------------------------------
# Footer
# ------------------------------
st.markdown(
    """
<div class="footer-note">
<b>Disclaimer:</b> This tool aims to cover a very broad range of algebra and calculus tasks via symbolic math. Some problems may not have closed forms or may require numeric initialization. If you hit a case that doesn't work, try the numeric methods or adjust assumptions.
</div>
    """,
    unsafe_allow_html=True,
)

import ast
import warnings
import hmac
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

warnings.filterwarnings("ignore")

OUTPUTS = {
    "GDP_growth": "GDP_growth",
    "Inflation_rate": "Inflation_rate",
    "Real_wage": "Real_wage",
    "Labor_productivity": "Labor_productivity",
    "Exchange_rate_PHP_to_USD": "Exchange_rate_PHP_to_USD",
}
INDEPENDENTS = [
    "Private_investment", "Public_investment", "Employment_rate", "HCI",
    "Capital_formation", "Real_wage", "GNI", "Labor_productivity",
    "GDP_growth", "Gov_exp", "Exchange_rate_PHP_to_USD", "Interest_rate",
    "Inflation_rate", "CSPI", "Business_confidence",
]
FORECAST_VARIABLES = [
    "GDP_growth", "Inflation_rate", "Real_wage", "Labor_productivity",
    "Exchange_rate_PHP_to_USD", "Capital_formation", "Business_confidence",
    "Private_investment", "Public_investment", "Employment_rate", "HCI",
    "GNI", "Gov_exp", "Interest_rate", "CSPI",
]

def normalise(value):
    return "".join(c.lower() for c in str(value) if c.isalnum())

def resolve_columns(frame):
    lookup = {normalise(c): c for c in frame.columns}
    aliases = {
        "Exchange_rate_PHP_to_USD": ["exchangerate", "exchangeratephptousd"],
        "Public_investment": ["publicinvestmentt"],
    }
    resolved = {}
    for name in set(INDEPENDENTS) | set(OUTPUTS.values()):
        candidates = [name] + aliases.get(name, [])
        resolved[name] = next((lookup[normalise(c)] for c in candidates if normalise(c) in lookup), None)
    return resolved

def parse_years(frame, column):
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values.astype("Int64").astype(str), format="%Y", errors="coerce")
    return pd.to_datetime(values, errors="coerce")

def clean_data(frame, date_column, fill_missing=True):
    result = frame.copy()
    result[date_column] = parse_years(result, date_column)
    result = result.dropna(subset=[date_column]).sort_values(date_column).set_index(date_column)
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.select_dtypes(include=[np.number])
    if fill_missing:
        result = result.interpolate(limit_direction="both").ffill().bfill()
    return result

def safe_log(series):
    return np.log(series.where(series > 0))

def filter_multicollinear_variables(frame, variables, threshold=0.95):
    ordered = []
    for variable in variables:
        if variable not in frame.columns:
            continue
        if not ordered:
            ordered.append(variable)
            continue
        correlations = []
        for existing in ordered:
            corr = frame[[variable, existing]].corr().iloc[0, 1]
            correlations.append(abs(float(corr)) if pd.notna(corr) else 0.0)
        if max(correlations, default=0.0) < threshold:
            ordered.append(variable)
    return ordered

def equation_features(frame, columns, keep_variables=None):
    keep_set = set(keep_variables) if keep_variables is not None else None

    def column(name):
        source = columns.get(name)
        if source is None:
            return pd.Series(np.nan, index=frame.index)
        if keep_set is not None and source not in keep_set:
            return pd.Series(np.nan, index=frame.index)
        return frame[source]

    result = pd.DataFrame(index=frame.index)
    result["eq_capital_formation"] = safe_log(column("Private_investment")) + safe_log(column("Public_investment"))
    result["eq_labor_productivity"] = column("Employment_rate") + column("HCI") + column("Capital_formation") + column("Real_wage") + column("Inflation_rate") + safe_log(column("GNI"))
    result["eq_gdp_growth"] = column("Labor_productivity").diff()
    result["eq_business_confidence"] = column("GDP_growth") + safe_log(column("Gov_exp")) + column("Interest_rate") + column("Inflation_rate") + safe_log(column("Exchange_rate_PHP_to_USD")) + safe_log(column("CSPI"))
    result["eq_private_investment"] = safe_log(column("Private_investment")).diff() + column("GDP_growth") + column("Business_confidence")
    return result.replace([np.inf, -np.inf], np.nan)

def evaluate_user_equation(expression, frame):
    expr = (expression or "").strip()
    if not expr:
        return None

    if expr.count("=") != 1:
        raise ValueError("Equation must contain exactly one '=' with the dependent variable on the left and expression on the right.")

    lhs, rhs = [part.strip() for part in expr.split("=", 1)]
    if not lhs or not rhs:
        raise ValueError("Equation must have both a dependent variable and a right-hand expression.")

    if not lhs.replace("_", "").isalnum():
        raise ValueError("Dependent variable name can contain letters, numbers, and underscores only.")

    try:
        parsed = ast.parse(rhs, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid syntax in equation: {exc.msg}") from exc

    allowed_funcs = {"log": np.log, "sqrt": np.sqrt, "abs": np.abs}
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )

    def validate(node):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
        if isinstance(node, ast.BinOp):
            validate(node.left)
            validate(node.right)
        elif isinstance(node, ast.UnaryOp):
            validate(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                raise ValueError("Only log(), sqrt(), and abs() are allowed in custom equations.")
            for arg in node.args:
                validate(arg)
            for keyword in node.keywords:
                validate(keyword.value)
        elif isinstance(node, ast.Name):
            if node.id not in set(frame.columns) | set(allowed_funcs):
                raise ValueError(f"Unknown variable or function: {node.id}")

    validate(parsed)

    local_context = {name: frame[name] for name in frame.columns}
    local_context.update(allowed_funcs)
    result = eval(compile(parsed, "<custom_equation>", "eval"), {"__builtins__": {}}, local_context)
    output = result if isinstance(result, pd.Series) else pd.Series(result, index=frame.index)
    output = output.replace([np.inf, -np.inf], np.nan)
    return lhs, output

def metric_values(actual, predicted, training):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    scale = np.mean(np.abs(np.diff(np.asarray(training, dtype=float))))
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    denominator = np.where(np.abs(actual) < 1e-8, 1e-8, np.abs(actual))
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE": float(np.mean(np.abs((actual - predicted) / denominator)) * 100),
        "MASE": float(np.mean(np.abs(actual - predicted)) / scale),
    }

def arima_forecast(train, steps):
    return np.column_stack([ARIMA(train[column], order=(1, 1, 1), trend="t").fit().forecast(steps) for column in train.columns])

def var_forecast(train, steps):
    fitted = VAR(train).fit(maxlags=min(4, max(1, len(train) // 10)), ic="aic", trend="c")
    return fitted.forecast(train.values[-fitted.k_ar:], steps)

def vecm_forecast(train, steps):
    fitted = VECM(train, k_ar_diff=1, coint_rank=min(1, len(train.columns) - 1), deterministic="co").fit()
    return fitted.predict(steps=steps)

def garch_forecast(train, steps):
    from arch import arch_model
    forecasts = []
    for column in train.columns:
        changes = train[column].diff().dropna()
        fitted = arch_model(changes, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
        mean_changes = fitted.forecast(horizon=steps).mean.iloc[-1].to_numpy()
        forecasts.append(train[column].iloc[-1] + np.cumsum(mean_changes))
    return np.column_stack(forecasts)

def xgb_features(frame, targets, exogenous, equations, index, history):
    values = {}
    for lag in (1, 2, 3):
        for target in targets:
            values[f"{target}_lag{lag}"] = history[target].iloc[-lag]
    source = frame.loc[index] if index in frame.index else frame.iloc[-1]
    for column in exogenous + equations:
        values[column] = source.get(column, frame[column].iloc[-1] if column in frame else 0)
    return values

def xgboost_forecast(frame, train, targets, exogenous, equations, steps, future=False):
    from xgboost import XGBRegressor
    rows, labels = [], {target: [] for target in targets}
    for position in range(3, len(train)):
        rows.append(xgb_features(frame, targets, exogenous, equations, train.index[position], train.iloc[:position]))
        for target in targets:
            labels[target].append(train.iloc[position][target])
    design = pd.DataFrame(rows).fillna(0)
    models = {
        target: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=2, verbosity=0).fit(design, labels[target])
        for target in targets
    }
    history = train.copy()
    indexes = frame.index[-steps:] if future else frame.index[len(train):len(train) + steps]
    predictions = []
    for index in indexes:
        row = pd.DataFrame([xgb_features(frame, targets, exogenous, equations, index, history)]).fillna(0)
        prediction = {target: float(models[target].predict(row)[0]) for target in targets}
        predictions.append([prediction[target] for target in targets])
        history = pd.concat([history, pd.DataFrame([prediction], index=[index])])
    return np.asarray(predictions)

def select_variance_stable_vector_data(frame, targets, equations, columns, threshold=0.95, max_equations=2):
    selected = list(targets)
    if not equations:
        return frame[selected]

    equation_columns = [column for column in equations if column in frame.columns and column not in targets]
    if not equation_columns:
        return frame[selected]

    candidate_sources = [source for source in columns.values() if source is not None and source not in targets]
    filtered_sources = filter_multicollinear_variables(frame, candidate_sources, threshold=threshold)
    filtered_equations = equation_features(frame, columns, keep_variables=set(filtered_sources))
    equation_candidates = [column for column in equation_columns if column in filtered_equations.columns and filtered_equations[column].notna().any()]

    if not equation_candidates:
        return pd.concat([frame[selected], filtered_equations.iloc[:, :1]], axis=1)

    kept = equation_candidates[:max_equations]
    if not kept:
        kept = [filtered_equations.columns[0]]
    return pd.concat([frame[selected], filtered_equations[kept]], axis=1)


def run_method(method, frame, train, targets, exogenous, equations, steps, use_equations_in_var_vecm=False, resolved_columns=None):
    if method in ("VAR", "VECM"):
        vector_data = train[targets]
        if use_equations_in_var_vecm and equations:
            resolved_columns = resolved_columns or {target: target for target in targets}
            vector_data = select_variance_stable_vector_data(train, targets, equations, resolved_columns)
        try:
            if method == "VAR":
                return var_forecast(vector_data, steps)[:, :len(targets)]
            return vecm_forecast(vector_data, steps)[:, :len(targets)]
        except Exception:
            if use_equations_in_var_vecm and equations:
                fallback = train[targets]
                if method == "VAR":
                    return var_forecast(fallback, steps)[:, :len(targets)]
                return vecm_forecast(fallback, steps)[:, :len(targets)]
            raise

    signal_columns = list(targets) + [column for column in equations if column not in targets]
    feature_set = train[signal_columns] if signal_columns else train[targets]

    if method == "ARIMA":
        return arima_forecast(feature_set, steps)[:, :len(targets)]
    if method == "GARCH":
        return garch_forecast(feature_set, steps)[:, :len(targets)]
    return xgboost_forecast(frame, train[targets], targets, exogenous, equations, steps)

def in_sample_predictions(frame, train, targets, exogenous, equations, method, use_equations_in_var_vecm=False, resolved_columns=None):
    if method == "XGBoost":
        from xgboost import XGBRegressor

        rows, labels = [], {target: [] for target in targets}
        for position in range(3, len(train)):
            rows.append(xgb_features(frame, targets, exogenous, equations, train.index[position], train.iloc[:position]))
            for target in targets:
                labels[target].append(train.iloc[position][target])
        design = pd.DataFrame(rows).fillna(0)
        models = {
            target: XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
                verbosity=0,
            ).fit(design, labels[target])
            for target in targets
        }
        predictions = np.full((len(train), len(targets)), np.nan)
        for position in range(3, len(train)):
            row = pd.DataFrame([
                xgb_features(frame, targets, exogenous, equations, train.index[position], train.iloc[:position])
            ]).fillna(0)
            predictions[position] = [float(models[target].predict(row)[0]) for target in targets]
        return predictions

    if method in ("VAR", "VECM"):
        vector_data = train[targets]
        if use_equations_in_var_vecm and equations:
            resolved_columns = resolved_columns or {target: target for target in targets}
            vector_data = select_variance_stable_vector_data(train, targets, equations, resolved_columns)
        try:
            if method == "VAR":
                fitted = VAR(vector_data).fit(maxlags=min(4, max(1, len(vector_data) // 10)), ic="aic", trend="c")
                values = fitted.fittedvalues
                start = fitted.k_ar
            else:
                fitted = VECM(
                    vector_data,
                    k_ar_diff=1,
                    coint_rank=min(1, len(vector_data.columns) - 1),
                    deterministic="co",
                ).fit()
                values = fitted.fittedvalues
                start = len(vector_data) - len(values)
        except (ValueError, np.linalg.LinAlgError):
            vector_data = train[targets]
            if method == "VAR":
                fitted = VAR(vector_data).fit(maxlags=min(4, max(1, len(vector_data) // 10)), ic="aic", trend="c")
                values = fitted.fittedvalues
                start = fitted.k_ar
            else:
                fitted = VECM(
                    vector_data,
                    k_ar_diff=1,
                    coint_rank=min(1, len(vector_data.columns) - 1),
                    deterministic="co",
                ).fit()
                values = fitted.fittedvalues
                start = len(vector_data) - len(values)
        predictions = np.full((len(train), len(targets)), np.nan)
        predictions[start:start + len(values)] = np.asarray(values)[:, :len(targets)]
        return predictions

    signal_columns = list(targets) + [column for column in equations if column not in targets]
    feature_set = train[signal_columns] if signal_columns else train[targets]
    predictions = np.full((len(train), len(targets)), np.nan)
    if method == "ARIMA":
        for position, column in enumerate(feature_set.columns[:len(targets)]):
            fitted = ARIMA(feature_set[column], order=(1, 1, 1), trend="t").fit()
            values = np.asarray(fitted.fittedvalues, dtype=float)
            predictions[-len(values):, position] = values
        return predictions
    if method == "GARCH":
        from arch import arch_model

        for position, column in enumerate(feature_set.columns[:len(targets)]):
            changes = feature_set[column].diff().dropna()
            fitted = arch_model(changes, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
            mean_change = float(fitted.params.get("Const", 0.0))
            predictions[1:, position] = feature_set[column].iloc[:-1].to_numpy() + mean_change
        return predictions
    raise ValueError(f"Unsupported method: {method}")

def in_sample_metric_values(actual, predicted, training):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        raise ValueError("The model did not produce valid in-sample predictions.")
    return metric_values(actual[valid], predicted[valid], training)

def run_prediction_mode(prediction_mode, methods, model_frame, train, test, targets, exogenous, equation_columns, target_labels, use_equations_in_var_vecm, resolved):
    predictions, errors = {}, {}
    for method in methods:
        try:
            if prediction_mode == "Out-of-sample backtest":
                predictions[method] = run_method(
                    method,
                    model_frame,
                    train,
                    targets,
                    exogenous,
                    equation_columns,
                    len(test),
                    use_equations_in_var_vecm=use_equations_in_var_vecm,
                    resolved_columns=resolved,
                )
                errors[method] = pd.DataFrame(
                    [
                        metric_values(test[target], predictions[method][:, position], train[target])
                        for position, target in enumerate(targets)
                    ],
                    index=target_labels,
                )
            else:
                predictions[method] = in_sample_predictions(
                    model_frame,
                    model_frame,
                    targets,
                    exogenous,
                    equation_columns,
                    method,
                    use_equations_in_var_vecm=use_equations_in_var_vecm,
                    resolved_columns=resolved,
                )
                errors[method] = pd.DataFrame(
                    [
                        in_sample_metric_values(model_frame[target], predictions[method][:, position], model_frame[target])
                        for position, target in enumerate(targets)
                    ],
                    index=target_labels,
                )
        except Exception as error:
            st.warning(f"{method} could not be fitted for {prediction_mode}: {error}")
    return predictions, errors

st.set_page_config(page_title="Macroeconomic Forecasting Lab", layout="wide")

st.markdown(
    """
    <style>
        div[data-baseweb="tag"],
        [role="option"][aria-selected="true"],
        [role="listbox"] [aria-selected="true"],
        [aria-selected="true"] {
            background-color: #1ed760 !important;
            color: #0b0f0d !important;
            border: 1px solid rgba(30, 215, 96, 0.7) !important;
        }
        div[data-baseweb="tag"] span,
        [role="option"][aria-selected="true"] span,
        [role="listbox"] [aria-selected="true"] span,
        [aria-selected="true"] span {
            color: #0b0f0d !important;
        }
        div[data-baseweb="select"] [role="combobox"],
        div[data-baseweb="select"] {
            border-color: rgba(30, 215, 96, 0.7) !important;
            box-shadow: 0 0 0 1px rgba(30, 215, 96, 0.5) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/1/1c/Philippine_Institute_for_Development_Studies_%28PIDS%29.svg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original"

def render_app_title():
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 1.5rem; margin: 0.5rem 0 1rem; width: 100%;">
            <div style="display: flex; flex-direction: column; justify-content: center; min-width: 0; flex: 3; padding-right: 0.5 rem;">
                <div style="font-size: 4.2rem; line-height: 0.9; font-weight: 800; margin: 0; letter-spacing: -0.06em;">Macroeconomic</div>
                <div style="font-size: 4.2rem; line-height: 0.9; font-weight: 800; margin: 0; letter-spacing: -0.06em;">Forecasting Lab</div>
            </div>
            <div style="flex: 1; display: flex; justify-content: flex-end; align-items: center;">
                <img src="{LOGO_URL}" alt="App logo" style="width: 170px; height: 170px; object-fit: contain;">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def require_password():
    configured_password = st.secrets.get("APP_PASSWORD")
    if not configured_password:
        st.error("APP_PASSWORD is not configured in Streamlit Secrets.")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    render_app_title()
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if hmac.compare_digest(password, str(configured_password)):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()

require_password()
render_app_title()
st.caption("Equation-informed comparison of ARIMA, VECM, VAR, GARCH and XGBoost")

csv_path = Path(__file__).with_name("R1_model.csv")
metadata_path = Path(__file__).with_name("Metadata.csv")
if not csv_path.exists():
    st.error(f"Required input file not found: {csv_path.name}")
    st.stop()

raw = pd.read_csv(csv_path)
with st.sidebar:
    st.header("Forecast settings")
    st.download_button(
        "Download historical data",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
        use_container_width=True,
    )
    if metadata_path.exists():
        st.download_button(
            "Download metadata",
            data=metadata_path.read_bytes(),
            file_name=metadata_path.name,
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.error(f"Metadata file not found: {metadata_path.name}")
    with st.form("forecast_settings"):
        st.caption("Choose variables, methods, and horizon, then submit the settings.")
        forecast_outputs = st.multiselect(
            "Forecast outputs",
            FORECAST_VARIABLES,
            default=[],
        )
        forecast_inputs = st.multiselect(
            "Forecast inputs",
            FORECAST_VARIABLES,
            default=[],
        )
        st.markdown(
            "<div style='display:flex; align-items:center; gap:0.5rem;'>"
            "<span>Custom equation (optional)</span>"
            "<span title='Optional: add a custom equation only if you want to include one in the forecast models. The variables are case sensitive so please copy the name of variables written in the editable data preview. Else, error will prompt' style='display:inline-flex; align-items:center; justify-content:center; width:1.2rem; height:1.2rem; border-radius:50%; background:#2d7df6; color:white; font-size:0.8rem; font-weight:700; cursor:help;'>?</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Example: GDP_growth = Inflation_rate + Employment_rate. Use one dependent variable on the left and allowed math on the right.")
        max_custom_equations = 10
        if "custom_equation_count" not in st.session_state:
            st.session_state["custom_equation_count"] = 1
        st.session_state["custom_equation_count"] = min(
            st.session_state["custom_equation_count"], max_custom_equations
        )
        custom_equations = [
            st.text_input(
                f"Custom equation {index}",
                value="",
                placeholder="e.g. GDP_growth = Inflation_rate + Employment_rate",
                key=f"custom_equation_{index}",
            )
            for index in range(1, st.session_state["custom_equation_count"] + 1)
        ]
        if st.session_state["custom_equation_count"] < max_custom_equations:
            add_equation = st.form_submit_button("Add equation", use_container_width=True)
        else:
            add_equation = False
            st.caption("Maximum of 10 custom equations reached.")
        selected_methods = st.multiselect(
            "Methods to compare",
            ["ARIMA", "VAR", "VECM", "GARCH", "XGBoost"],
            default=[],
        )
        use_equations_in_var_vecm = True
        forecast_horizon = st.number_input("Forecast horizon (years)", min_value=1, max_value=10, value=5, step=1)
        st.markdown(
            "<div style='display:flex; align-items:center; gap:0.5rem; margin-top:0.25rem;'>"
            "<span style='font-size:1.1rem; font-weight:700;'>Backtest share</span>"
            "<span title='Share of the dataset used for backtesting; the rest is used for training. Example: 0.20 means 20% held out for validation.' style='display:inline-flex; align-items:center; justify-content:center; width:1.2rem; height:1.2rem; border-radius:50%; background:#2d7df6; color:white; font-size:0.8rem; font-weight:700; cursor:help;'>?</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        test_fraction = st.slider("", 0.1, 0.4, 0.2, 0.05, label_visibility="collapsed")
        prediction_mode = st.radio(
            "Prediction mode",
            ["Out-of-sample backtest", "In-sample fit", "Both"],
            index=0,
            help="Out-of-sample holds out the latest observations. In-sample fits each model on the full historical data. Both shows results from both modes.",
        )
        compare_prediction_modes = st.checkbox(
            "Compare in-sample and out-of-sample",
            value=False,
            help="Run both modes and compare their metrics for the same targets and methods.",
        )
        run_comparison = st.form_submit_button("Run comparison", type="primary", use_container_width=True)

if add_equation:
    st.session_state["custom_equation_count"] = min(
        st.session_state["custom_equation_count"] + 1, max_custom_equations
    )
    st.rerun()

resolved = resolve_columns(raw)
missing = [name for name, source in resolved.items() if source is None]
if missing:
    st.error("Missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

date_column = "Year" if "Year" in raw.columns else raw.columns[0]
data = clean_data(raw, date_column)
preview_data = clean_data(raw, date_column, fill_missing=False)
input_columns = [column for column in preview_data.columns]
preview = preview_data.loc[preview_data.index.year.isin([2024, 2025, 2026, 2027]), input_columns].copy()
preview.index = preview.index.year
preview.index.name = "Year"
if preview.empty:
    st.error("The CSV must contain rows for 2024, 2025, 2026, and 2027.")
    st.stop()

st.header("Editable input and output data preview")
st.caption("Edit any of the 16 non-date variables for 2024-2027. Year is shown only as the row index.")
edited_preview = st.data_editor(preview, num_rows="fixed", use_container_width=True, key="input_preview")
for year in edited_preview.index:
    row_mask = data.index.year == int(year)
    for column in input_columns:
        value = pd.to_numeric(edited_preview.loc[year, column], errors="coerce")
        if pd.notna(value):
            data.loc[row_mask, column] = value

if not run_comparison and "backtest_results" not in st.session_state:
    st.info("Choose your forecast settings, then click Run comparison in the sidebar.")
    st.stop()

if not forecast_outputs or not forecast_inputs or not selected_methods:
    st.warning("Select at least one output, input, and method in Forecast settings.")
    st.stop()

compare_prediction_modes = compare_prediction_modes or prediction_mode == "Both"

def build_equation_frame(frame, resolved, custom_equations):
    equations = equation_features(frame,resolved)

    for index, expression in enumerate(custom_equations, start=1):
        if not expression or not expression.strip():
            continue
        try:
            lhs, rhs_result = evaluate_user_equation(expression, frame)
            equation_name = f"eq_custom_{index}_{lhs}"
            equations[equation_name] = rhs_result
        except ValueError as exc:
            st.warning(
                f"Custom equation {index} is invalid: {exc}"
            )

    return equations

equations = build_equation_frame(
    data,
    resolved,
    custom_equations,
)

model_frame = pd.concat([data, equations], axis=1)
targets = [resolved[name] for name in forecast_outputs]
target_labels = forecast_outputs
equation_columns = list(equations.columns)
exogenous = [resolved[name] for name in forecast_inputs]
exogenous = [column for column in exogenous if column in model_frame.columns and column not in targets]
feature_columns = exogenous + equation_columns

if len(model_frame) < 30:
    st.error("At least 30 observations are recommended.")
    st.stop()

split = max(15, int(len(model_frame) * (1 - test_fraction)))
train, test = model_frame.iloc[:split], model_frame.iloc[split:]

cached_results = st.session_state.get("backtest_results", {})
cached_mode = cached_results.get("mode")
cached_compare = cached_results.get("compare_prediction_modes", False)
if run_comparison or "backtest_results" not in st.session_state or cached_mode != prediction_mode or cached_compare != compare_prediction_modes:
    predictions, errors = {}, {}
    selected_prediction_mode = "Out-of-sample backtest" if prediction_mode == "Both" else prediction_mode
    evaluation_label = "backtest" if selected_prediction_mode == "Out-of-sample backtest" else "in-sample fit"
    with st.spinner(f"Fitting models and running the {evaluation_label}..."):
        try:
            predictions, errors = run_prediction_mode(
                selected_prediction_mode,
                selected_methods,
                model_frame,
                train,
                test,
                targets,
                exogenous,
                equation_columns,
                target_labels,
                use_equations_in_var_vecm,
                resolved,
            )
        except Exception as error:
            st.warning(f"{selected_prediction_mode} could not be fitted: {error}")
    if not errors:
        st.stop()
    st.session_state["backtest_results"] = {
        "predictions": predictions,
        "errors": errors,
        "target_labels": target_labels,
        "mode": prediction_mode,
        "compare_prediction_modes": compare_prediction_modes,
    }
else:
    predictions = st.session_state["backtest_results"]["predictions"]
    errors = st.session_state["backtest_results"]["errors"]
    target_labels = st.session_state["backtest_results"]["target_labels"]

if compare_prediction_modes:
    comparison_errors = {}
    comparison_modes = ["In-sample fit", "Out-of-sample backtest"]
    with st.spinner("Comparing in-sample and out-of-sample results..."):
        for mode in comparison_modes:
            _, comparison_errors[mode] = run_prediction_mode(
                mode,
                selected_methods,
                model_frame,
                train,
                test,
                targets,
                exogenous,
                equation_columns,
                target_labels,
                use_equations_in_var_vecm,
                resolved,
            )

if not errors:
    st.stop()

display_prediction_mode = "Out-of-sample backtest" if prediction_mode == "Both" else prediction_mode
if not compare_prediction_modes:
    st.header("Both prediction results" if prediction_mode == "Both" else f"{prediction_mode} results")
    if prediction_mode == "Both":
        st.caption("Both prediction modes are shown below. The out-of-sample results use the latest observations as a holdout set.")
    elif display_prediction_mode == "Out-of-sample backtest":
        st.caption("Models are trained on the earlier observations and evaluated on the held-out latest observations.")
    else:
        st.caption("Models are fitted on the full historical dataset and evaluated on the same observations; these scores are descriptive, not validation scores.")
    metric_name = st.selectbox("Metric to rank", ["RMSE", "MAE", "MAPE", "MASE"], index=0)
    ranking = pd.DataFrame({method: result[metric_name] for method, result in errors.items()}, index=target_labels)
    st.dataframe(ranking.style.format("{:.4f}"), use_container_width=True)

in_sample_only = not compare_prediction_modes and prediction_mode == "In-sample fit"
best_errors = comparison_errors["Out-of-sample backtest"] if compare_prediction_modes else errors
best = pd.DataFrame(index=target_labels)
for metric in ["RMSE", "MAE", "MAPE", "MASE"]:
    scores = pd.DataFrame({method: result[metric] for method, result in best_errors.items()}, index=target_labels)
    if in_sample_only:
        best[f"2nd best model by {metric}"] = scores.apply(
            lambda row: row.dropna().sort_values().index[1]
            if len(row.dropna()) > 1
            else (row.dropna().index[0] if len(row.dropna()) else None),
            axis=1,
        )
    else:
        best[f"Best model by {metric}"] = scores.idxmin(axis=1)
best.insert(0, "Target", best.index)
if in_sample_only:
    st.subheader(
        "2nd best forecasting method",
        help="XGBoost is always the best in-sample model because it is the most flexible and has the lowest training error. This table shows the second-best model instead.",
    )
else:
    st.subheader("Best forecasting method (out-of-sample)" if compare_prediction_modes else "Best forecasting method")
st.dataframe(best.reset_index(drop=True), hide_index=True, use_container_width=True)

if compare_prediction_modes:
    st.header("In-sample vs out-of-sample comparison")
    st.subheader("Prediction error comparison")
    st.caption("The table compares the selected error metric. Positive differences mean the out-of-sample error is higher than the in-sample error.")
    comparison_metric = st.selectbox("Comparison metric", ["RMSE", "MAE", "MAPE", "MASE"], index=0)
    for mode in ["In-sample fit", "Out-of-sample backtest"]:
        mode_ranking = pd.DataFrame(
            {
                method: result[comparison_metric]
                for method, result in comparison_errors[mode].items()
            },
            index=target_labels,
        )
        st.subheader(mode)
        st.dataframe(mode_ranking.style.format("{:.4f}"), use_container_width=True)
    comparison_rows = []
    for target in target_labels:
        for method in selected_methods:
            if method not in comparison_errors["In-sample fit"] or method not in comparison_errors["Out-of-sample backtest"]:
                continue
            in_sample_score = comparison_errors["In-sample fit"][method].loc[target, comparison_metric]
            out_of_sample_score = comparison_errors["Out-of-sample backtest"][method].loc[target, comparison_metric]
            difference = out_of_sample_score - in_sample_score
            percent_difference = (difference / abs(in_sample_score) * 100) if abs(in_sample_score) > 1e-12 else np.nan
            comparison_rows.append({
                "Target": target,
                "Method": method,
                "In-sample": in_sample_score,
                "Out-of-sample": out_of_sample_score,
                "Difference (out - in)": difference,
                "% Difference": percent_difference,
            })
    st.dataframe(
        pd.DataFrame(comparison_rows).style.format({
            "In-sample": "{:.4f}",
            "Out-of-sample": "{:.4f}",
            "Difference (out - in)": "{:.4f}",
            "% Difference": "{:.2f}%",
        }),
        hide_index=True,
        use_container_width=True,
    )
forecast_years = list(range(2026, 2026 + int(forecast_horizon)))
forecast_index = pd.to_datetime([f"{year}-12-31" for year in forecast_years])
future_base = pd.DataFrame([data.loc[data.index.year == 2027].iloc[-1].to_dict()] * len(forecast_years), index=forecast_index).reindex(columns=data.columns)
future_equations=build_equation_frame(future_base,resolved,custom_equations)
future_frame = pd.concat([future_base, equation_features(future_base, resolved)], axis=1)
forecast_frame = pd.concat([model_frame, future_frame])

def build_prediction_table(prediction_values, prediction_index):
    forecast_table = pd.DataFrame(index=prediction_index)
    forecast_table.index.name = "Year"
    for position, target in enumerate(target_labels):
        for method, values in prediction_values.items():
            forecast_table[(target, method)] = values[:, position]
    forecast_table.columns = pd.MultiIndex.from_tuples(forecast_table.columns)
    return forecast_table

def restrict_to_forecast_horizon(prediction_values, prediction_index):
    years = np.asarray(prediction_index, dtype=int)
    mask = np.isin(years, forecast_years)
    return {
        method: values[mask]
        for method, values in prediction_values.items()
    }, years[mask]

def restrict_to_post_sample_horizon(prediction_values):
    years = np.asarray(forecast_years, dtype=int)
    mask = years >= 2028
    return {
        method: values[mask]
        for method, values in prediction_values.items()
    }, years[mask]

def combine_prediction_values(first_values, first_years, second_values, second_years):
    methods = [method for method in first_values if method in second_values]
    combined_values = {
        method: np.vstack([first_values[method], second_values[method]])
        for method in methods
    }
    return combined_values, np.concatenate([first_years, second_years])

def future_predictions_for(training_frame, label):
    results = {}
    for method in selected_methods:
        try:
            results[method] = run_method(
                method,
                forecast_frame,
                training_frame,
                targets,
                exogenous,
                equation_columns,
                len(forecast_years),
                use_equations_in_var_vecm=use_equations_in_var_vecm,
                resolved_columns=resolved,
            )
        except Exception as error:
            st.warning(f"{method} {label} forecast failed: {error}")
    return results

def in_sample_display_predictions():
    fitted_predictions = {
        method: in_sample_predictions(
            model_frame,
            model_frame,
            targets,
            exogenous,
            equation_columns,
            method,
            use_equations_in_var_vecm=use_equations_in_var_vecm,
            resolved_columns=resolved,
        )
        for method in selected_methods
    }
    future_predictions = future_predictions_for(model_frame, "in-sample")
    observed_years = model_frame.index.year.to_numpy()
    display_predictions = {}
    for method in selected_methods:
        if method not in fitted_predictions or method not in future_predictions:
            continue
        fitted_values = fitted_predictions[method]
        future_values = future_predictions[method]
        values = np.full((len(forecast_years), len(targets)), np.nan)
        for position, year in enumerate(forecast_years):
            observed_positions = np.flatnonzero(observed_years == year)
            if len(observed_positions):
                values[position] = fitted_values[observed_positions[-1]]
            else:
                future_position = year - forecast_years[0]
                values[position] = future_values[future_position]
        display_predictions[method] = values
    return display_predictions

def render_forecast_table(title, caption, prediction_values, filename, prediction_index=None):
    st.subheader(title)
    st.caption(caption)
    if not prediction_values:
        st.warning("No forecast estimates were produced for this prediction mode.")
        return
    forecast_table = build_prediction_table(
        prediction_values,
        forecast_years if prediction_index is None else prediction_index,
    )
    st.dataframe(forecast_table.style.format("{:.4f}"), use_container_width=True)
    download = forecast_table.copy()
    download.columns = [f"{target}_{method}" for target, method in download.columns]
    st.download_button(
        "Download forecast table",
        download.reset_index().to_csv(index=False),
        filename,
        "text/csv",
        key=f"download_{filename}",
    )

st.header(f"{forecast_years[0]}-{forecast_years[-1]} forecasts")
if prediction_mode == "Both":
    in_sample_future_predictions = in_sample_display_predictions()
    render_forecast_table(
        f"In-sample forecast estimates ({forecast_years[0]}-{forecast_years[-1]})",
        "Forecasts from models fitted using the full historical dataset.",
        in_sample_future_predictions,
        "in_sample_forecasts_{forecast_years[0]}_{forecast_years[-1]}.csv",
    )
    out_of_sample_predictions = future_predictions_for(model_frame, "out-of-sample")
    render_forecast_table(
        f"Out-of-sample forecast estimates ({forecast_years[0]}-{forecast_years[-1]}, backtest share: {test_fraction:.0%})",
        "The displayed forecast uses the same full-history projection logic across all years. The retained chronological backtest remains available above for validation.",
        out_of_sample_predictions,
        "out_of_sample_forecasts_{forecast_years[0]}_{forecast_years[-1]}.csv",
    )
else:
    if prediction_mode == "In-sample fit":
        future_predictions = in_sample_display_predictions()
        if future_predictions:
            forecast_table = build_prediction_table(future_predictions, forecast_years)
            st.dataframe(forecast_table.style.format("{:.4f}"), use_container_width=True)
            download = forecast_table.copy()
            download.columns = [f"{target}_{method}" for target, method in download.columns]
            st.download_button("Download forecasts as CSV", download.reset_index().to_csv(index=False), f"macroeconomic_forecasts_{forecast_years[0]}_{forecast_years[-1]}.csv", "text/csv")
    else:
        st.subheader(f"Out-of-sample forecast estimates (2026-2030, backtest share: {test_fraction:.0%})")
        st.caption("The displayed forecast uses the same full-history projection logic across all five years. The retained chronological backtest remains available above for validation.")
        out_of_sample_predictions = future_predictions_for(model_frame, "out-of-sample")
        if out_of_sample_predictions:
            forecast_table = build_prediction_table(out_of_sample_predictions, forecast_years)
            st.dataframe(forecast_table.style.format("{:.4f}"), use_container_width=True)
        else:
            st.info("No out-of-sample forecast values are available for the 2026-2030 horizon.")
    
if prediction_mode == "Both":
    forecast_difference_rows = []
    for target_position, target in enumerate(target_labels):
        for method in selected_methods:
            if method not in in_sample_future_predictions or method not in out_of_sample_predictions:
                continue
            in_sample_values = in_sample_future_predictions[method][:, target_position]
            out_of_sample_values = out_of_sample_predictions[method][:, target_position]
            valid = np.isfinite(in_sample_values) & np.isfinite(out_of_sample_values)
            if not valid.any():
                continue
            differences = out_of_sample_values[valid] - in_sample_values[valid]
            denominator = np.where(np.abs(in_sample_values[valid]) > 1e-12, np.abs(in_sample_values[valid]), np.nan)
            percent_deviations = differences / denominator * 100
            forecast_difference_rows.append({
                "Target": target,
                "Method": method,
                "Mean absolute forecast difference": float(np.mean(np.abs(differences))),
                "Mean signed forecast difference (out - in)": float(np.mean(differences)),
                "Mean absolute percentage difference": float(np.nanmean(np.abs(percent_deviations))),
            })
    if forecast_difference_rows:
        st.header(f"{forecast_years[0]}_{forecast_years[-1]} forecast difference")
        st.caption("This calculation compares future forecast estimates from full-history in-sample models with forecasts trained only through the pre-holdout period. It is separate from the original out-of-sample backtest.")
        st.dataframe(
            pd.DataFrame(forecast_difference_rows).style.format({
                "Mean absolute forecast difference": "{:.4f}",
                "Mean signed forecast difference (out - in)": "{:.4f}",
                "Mean absolute percentage difference": "{:.2f}%",
            }),
            hide_index=True,
            use_container_width=True,
        )





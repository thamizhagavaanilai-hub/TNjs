import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import warnings
import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from ecmwf.opendata import Client

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER


# ================================================================
# TAMIL NADU ECMWF IFS RAINFALL - MOBILE WEB APP
# ================================================================

st.set_page_config(
    page_title="Tamil Nadu ECMWF Rainfall",
    page_icon="🌧️",
    layout="centered"
)

warnings.filterwarnings("ignore")

# ------------------------------------------------
# ECMWF SETTINGS
# ------------------------------------------------
MODEL = "ifs"
RESOLUTION = "0p25"
PARAMETER = "tp"
STREAM = "oper"
FORECAST_TYPE = "fc"
RUN_TIME = 0
RUN_DATE = None

START_STEP = 0
STEP_INTERVAL = 3

MIN_LON = 76.0
MAX_LON = 80.5
MIN_LAT = 8.0
MAX_LAT = 13.7


# ================================================================
# HELPERS
# ================================================================

def find_district_column(columns):
    keywords = [
        "dist_name", "district_name", "districtname", "dtname",
        "distname", "district_n", "dist_n", "district", "dist"
    ]

    for keyword in keywords:
        for column in columns:
            col = str(column).lower().strip()
            if col == keyword or keyword in col:
                if "code" not in col and "id" not in col:
                    return column
    return None


def find_taluk_column(columns):
    keywords = [
        "taluk_name", "talukname", "tkname", "taluk",
        "sub_dist", "subdistrict", "sub_dist_name",
        "block", "subdistname"
    ]

    for keyword in keywords:
        for column in columns:
            col = str(column).lower().strip()
            if keyword in col:
                if "code" not in col and "id" not in col:
                    return column
    return None


def clean_name(value):
    if value is None or str(value).strip() == "":
        return "Unknown"
    return str(value).strip().title()


def download_ecmwf(forecast_days, output_dir, status):
    forecast_hours = forecast_days * 24
    end_step = forecast_hours

    steps = list(range(START_STEP, end_step + 1, STEP_INTERVAL))

    grib_file = output_dir / f"ecmwf_ifs_tn_{forecast_hours}h_tp.grib2"

    status.write("Connecting to ECMWF Open Data...")

    client = Client(
        source="azure",
        model=MODEL,
        resol=RESOLUTION,
        infer_stream_keyword=False
    )

    kwargs = dict(
        time=RUN_TIME,
        stream=STREAM,
        type=FORECAST_TYPE,
        step=steps,
        param=PARAMETER,
        target=str(grib_file)
    )

    if RUN_DATE is not None:
        kwargs["date"] = RUN_DATE

    client.retrieve(**kwargs)

    model_datetime = datetime.utcnow().replace(
        hour=RUN_TIME,
        minute=0,
        second=0,
        microsecond=0
    )

    return grib_file, model_datetime


def read_ecmwf(grib_file, status):
    status.write("Reading ECMWF GRIB data...")

    ds = xr.open_dataset(
        grib_file,
        engine="cfgrib"
    )

    if "tp" not in ds:
        ds.close()
        raise KeyError("ECMWF total precipitation variable 'tp' was not found.")

    tp = ds["tp"]

    if "latitude" in tp.coords:
        lat = np.asarray(tp["latitude"].values, dtype=float)
    elif "lat" in tp.coords:
        lat = np.asarray(tp["lat"].values, dtype=float)
    else:
        ds.close()
        raise KeyError("Latitude coordinate not found.")

    if "longitude" in tp.coords:
        lon = np.asarray(tp["longitude"].values, dtype=float)
    elif "lon" in tp.coords:
        lon = np.asarray(tp["lon"].values, dtype=float)
    else:
        ds.close()
        raise KeyError("Longitude coordinate not found.")

    values = np.asarray(tp.values, dtype=float)

    if values.ndim == 3:
        accumulated = values[-1] - values[0]
    elif values.ndim == 2:
        accumulated = values
    else:
        ds.close()
        raise ValueError(f"Unexpected TP dimensions: {values.shape}")

    rainfall = accumulated * 1000.0
    rainfall = np.where(rainfall < 0, 0, rainfall)
    rainfall = np.nan_to_num(
        rainfall, nan=0.0, posinf=0.0, neginf=0.0
    )

    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    rainfall = rainfall[lat_order, :]

    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    rainfall = rainfall[:, lon_order]

    lon_mask = (lon >= MIN_LON) & (lon <= MAX_LON)
    lat_mask = (lat >= MIN_LAT) & (lat <= MAX_LAT)

    if not np.any(lon_mask):
        ds.close()
        raise ValueError("No longitude grid points found inside Tamil Nadu extent.")

    if not np.any(lat_mask):
        ds.close()
        raise ValueError("No latitude grid points found inside Tamil Nadu extent.")

    lon = lon[lon_mask]
    lat = lat[lat_mask]
    rainfall = rainfall[np.ix_(lat_mask, lon_mask)]

    ds.close()

    return lon, lat, rainfall


def create_interpolator(lon, lat, rainfall):
    return RegularGridInterpolator(
        (lat, lon),
        rainfall,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )


def load_villages(village_file):
    gdf = gpd.read_file(village_file)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    district_col = find_district_column(gdf.columns)
    taluk_col = find_taluk_column(gdf.columns)

    if district_col is None:
        raise KeyError(
            "District column was not found. Available columns: "
            + str(list(gdf.columns))
        )

    gdf[district_col] = gdf[district_col].apply(clean_name)

    if taluk_col:
        gdf[taluk_col] = gdf[taluk_col].apply(clean_name)

    return gdf, district_col, taluk_col


def calculate_village_rainfall(gdf, interpolator):
    gdf = gdf.copy()

    points = gdf.geometry.representative_point()

    coordinates = np.column_stack(
        (points.y.values, points.x.values)
    )

    rainfall = interpolator(coordinates)

    rainfall = np.asarray(rainfall, dtype=float)
    rainfall = np.nan_to_num(
        rainfall, nan=0.0, posinf=0.0, neginf=0.0
    )
    rainfall = np.maximum(rainfall, 0)

    gdf["rainfall_mm"] = np.round(rainfall, 1)

    return gdf


def get_category(mm):
    if mm > 100:
        return "Very Heavy"
    if mm >= 50:
        return "Heavy"
    if mm >= 20:
        return "Moderate"
    if mm >= 10:
        return "Light to Moderate"
    if mm >= 5:
        return "Light"
    if mm > 0:
        return "Very Light"
    return "No Rain"


def get_category_color(mm):
    if mm > 100:
        return "#7f1d1d"
    if mm >= 50:
        return "#991b1b"
    if mm >= 20:
        return "#b45309"
    if mm >= 10:
        return "#ca8a04"
    if mm >= 5:
        return "#166534"
    return "#475569"


def get_forecast_text(model_datetime, forecast_days):
    forecast_hours = forecast_days * 24

    start_time = model_datetime + timedelta(hours=START_STEP)
    end_time = model_datetime + timedelta(hours=forecast_hours)

    return (
        start_time.strftime("%d %b %Y %H:%M UTC")
        + " to "
        + end_time.strftime("%d %b %Y %H:%M UTC")
    )


# ================================================================
# WARNING PDF
# ================================================================

def create_warning_pdf(
    gdf, district_col, taluk_col, model_datetime,
    forecast_days, output_file
):
    forecast_hours = forecast_days * 24

    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        fontName="Helvetica-Bold"
    )

    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=8,
        leading=10
    )

    bold_style = ParagraphStyle(
        "Bold",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        fontName="Helvetica-Bold"
    )

    elements = []

    forecast_text = get_forecast_text(model_datetime, forecast_days)

    elements.append(
        Paragraph("TAMIL NADU RAINFALL WARNING REPORT", title_style)
    )
    elements.append(Spacer(1, 4))
    elements.append(
        Paragraph("ECMWF IFS MODEL GUIDANCE", subtitle_style)
    )
    elements.append(
        Paragraph(
            f"{forecast_days}-DAY / {forecast_hours}-HOUR CUMULATIVE RAINFALL",
            subtitle_style
        )
    )
    elements.append(
        Paragraph(
            f"FORECAST PERIOD: {forecast_text}",
            subtitle_style
        )
    )
    elements.append(Spacer(1, 12))

    district_peak = (
        gdf.groupby(district_col)["rainfall_mm"]
        .max()
        .reset_index()
    )

    district_peak.sort_values(
        "rainfall_mm",
        ascending=False,
        inplace=True
    )

    categories = [
        (
            "VERY HEAVY RAINFALL",
            district_peak[district_peak["rainfall_mm"] > 100],
            "#7f1d1d"
        ),
        (
            "HEAVY RAINFALL",
            district_peak[
                (district_peak["rainfall_mm"] >= 50)
                & (district_peak["rainfall_mm"] <= 100)
            ],
            "#991b1b"
        ),
        (
            "MODERATE RAINFALL",
            district_peak[
                (district_peak["rainfall_mm"] >= 20)
                & (district_peak["rainfall_mm"] < 50)
            ],
            "#b45309"
        )
    ]

    any_warning = False

    for category_title, category_df, text_color in categories:

        if category_df.empty:
            continue

        any_warning = True

        elements.append(
            Paragraph(f"<u>{category_title}</u>", heading_style)
        )
        elements.append(Spacer(1, 6))

        for _, district_row in category_df.iterrows():

            district_name = str(district_row[district_col])
            peak_mm = float(district_row["rainfall_mm"])

            district_header = Paragraph(
                f"<b>DISTRICT: {district_name.upper()}</b> "
                f"(Peak: {peak_mm:.1f} mm / {peak_mm / 10:.1f} cm)",
                bold_style
            )

            header_table = Table([[district_header]], colWidths=[530])

            header_table.setStyle(
                TableStyle([
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3)
                ])
            )

            elements.append(header_table)

            district_data = gdf[
                gdf[district_col] == district_name
            ].copy()

            if taluk_col:
                taluk_summary = (
                    district_data.groupby(taluk_col)["rainfall_mm"]
                    .max()
                    .reset_index()
                )

                taluk_summary.sort_values(
                    "rainfall_mm",
                    ascending=False,
                    inplace=True
                )

                table_data = [[
                    Paragraph("<b>Taluk Name</b>", bold_style),
                    Paragraph("<b>Peak Rainfall</b>", bold_style)
                ]]

                for _, taluk_row in taluk_summary.iterrows():
                    taluk_name = str(taluk_row[taluk_col])
                    mm = float(taluk_row["rainfall_mm"])

                    table_data.append([
                        Paragraph(taluk_name, normal_style),
                        Paragraph(
                            f"<font color='{text_color}'>"
                            f"<b>{mm:.1f} mm ({mm / 10:.1f} cm)</b>"
                            f"</font>",
                            normal_style
                        )
                    ])
            else:
                table_data = [
                    [
                        Paragraph("<b>District</b>", bold_style),
                        Paragraph("<b>Peak Rainfall</b>", bold_style)
                    ],
                    [
                        Paragraph(district_name, normal_style),
                        Paragraph(
                            f"<font color='{text_color}'>"
                            f"<b>{peak_mm:.1f} mm "
                            f"({peak_mm / 10:.1f} cm)</b></font>",
                            normal_style
                        )
                    ]
                ]

            rainfall_table = Table(
                table_data,
                colWidths=[330, 200],
                repeatRows=1
            )

            rainfall_table.setStyle(
                TableStyle([
                    (
                        "GRID", (0, 0), (-1, -1), 0.5,
                        colors.HexColor("#94a3b8")
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5)
                ])
            )

            elements.append(rainfall_table)
            elements.append(Spacer(1, 8))

    if not any_warning:
        elements.append(
            Paragraph(
                "No districts are forecast to receive 20 mm or more "
                "accumulated rainfall during the forecast period.",
                subtitle_style
            )
        )

    elements.append(Spacer(1, 15))
    elements.append(
        Paragraph(
            "<b>Note:</b> Rainfall values are ECMWF IFS model guidance "
            f"for the complete 0–{forecast_hours} hour accumulation period. "
            "They are not observed rainfall measurements.",
            normal_style
        )
    )

    doc.build(elements)


# ================================================================
# DISTRICT PDF
# ================================================================

def create_district_pdf(
    gdf, district_col, model_datetime,
    forecast_days, output_file
):
    forecast_hours = forecast_days * 24

    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=25,
        leftMargin=25,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DistrictTitle",
        parent=styles["Heading1"],
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )

    subtitle_style = ParagraphStyle(
        "DistrictSubtitle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )

    normal_style = ParagraphStyle(
        "DistrictNormal",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9
    )

    bold_style = ParagraphStyle(
        "DistrictBold",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9,
        fontName="Helvetica-Bold"
    )

    elements = []

    forecast_text = get_forecast_text(model_datetime, forecast_days)

    elements.append(
        Paragraph("TAMIL NADU DISTRICT RAINFALL SUMMARY", title_style)
    )
    elements.append(Spacer(1, 4))
    elements.append(
        Paragraph("ECMWF IFS MODEL GUIDANCE", subtitle_style)
    )
    elements.append(
        Paragraph(
            f"{forecast_days}-DAY / {forecast_hours}-HOUR CUMULATIVE RAINFALL",
            subtitle_style
        )
    )
    elements.append(
        Paragraph(
            f"FORECAST PERIOD: {forecast_text}",
            subtitle_style
        )
    )
    elements.append(Spacer(1, 12))

    district_stats = (
        gdf.groupby(district_col)["rainfall_mm"]
        .agg(
            average="mean",
            peak="max",
            minimum="min"
        )
        .reset_index()
    )

    for col in ["average", "peak", "minimum"]:
        district_stats[col] = district_stats[col].round(1)

    district_stats.sort_values(
        "average",
        ascending=False,
        inplace=True
    )

    table_data = [[
        Paragraph("<b>District</b>", bold_style),
        Paragraph("<b>Average</b>", bold_style),
        Paragraph("<b>Peak</b>", bold_style),
        Paragraph("<b>Minimum</b>", bold_style),
        Paragraph("<b>Category</b>", bold_style)
    ]]

    for _, row in district_stats.iterrows():

        district_name = str(row[district_col])
        average_mm = float(row["average"])
        peak_mm = float(row["peak"])
        minimum_mm = float(row["minimum"])

        category = get_category(average_mm)
        category_color = get_category_color(average_mm)

        table_data.append([
            Paragraph(district_name, bold_style),
            Paragraph(
                f"<b>{average_mm:.1f} mm</b><br/>"
                f"({average_mm / 10:.1f} cm)",
                normal_style
            ),
            Paragraph(
                f"{peak_mm:.1f} mm<br/>"
                f"({peak_mm / 10:.1f} cm)",
                normal_style
            ),
            Paragraph(f"{minimum_mm:.1f} mm", normal_style),
            Paragraph(
                f"<font color='{category_color}'>"
                f"<b>{category}</b></font>",
                normal_style
            )
        ])

    table = Table(
        table_data,
        colWidths=[125, 95, 95, 75, 135],
        repeatRows=1
    )

    table.setStyle(
        TableStyle([
            (
                "GRID", (0, 0), (-1, -1), 0.5,
                colors.HexColor("#94a3b8")
            ),
            (
                "BACKGROUND", (0, 0), (-1, 0),
                colors.HexColor("#e2e8f0")
            ),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "<b>Method:</b> District rainfall statistics are calculated "
            "from the rainfall values sampled at the village locations.",
            normal_style
        )
    )

    elements.append(Spacer(1, 6))

    elements.append(
        Paragraph(
            "<b>Note:</b> Values are ECMWF IFS model guidance "
            "and not observed rainfall.",
            normal_style
        )
    )

    doc.build(elements)


# ================================================================
# STREAMLIT USER INTERFACE
# ================================================================

st.title("🌧️ Tamil Nadu ECMWF Rainfall")
st.caption("Mobile-friendly ECMWF IFS cumulative rainfall forecast")

st.info(
    "Choose the forecast period, upload your Tamil Nadu village GeoJSON, "
    "then tap Generate Report."
)

forecast_days = st.selectbox(
    "Forecast period",
    options=[1, 2, 3, 4, 5, 6, 7],
    index=4,
    format_func=lambda x: f"{x} days ({x * 24} hours)"
)

village_upload = st.file_uploader(
    "Upload Tamil Nadu village GeoJSON",
    type=["geojson", "json"]
)

st.write(
    f"**ECMWF:** IFS | **Resolution:** {RESOLUTION} | "
    f"**Forecast:** {forecast_days * 24} hours"
)

generate = st.button(
    "🚀 Generate Rainfall Reports",
    type="primary",
    use_container_width=True
)

if generate:

    if village_upload is None:
        st.error("Please upload the Tamil Nadu village GeoJSON file.")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)
        village_path = temp_path / village_upload.name

        with open(village_path, "wb") as f:
            f.write(village_upload.getbuffer())

        progress = st.status(
            "Starting rainfall processing...",
            expanded=True
        )

        try:
            grib_file, model_datetime = download_ecmwf(
                forecast_days,
                temp_path,
                progress
            )

            progress.write("ECMWF download completed.")

            lon, lat, rainfall = read_ecmwf(
                grib_file,
                progress
            )

            progress.write("Creating rainfall interpolation...")

            interpolator = create_interpolator(
                lon, lat, rainfall
            )

            progress.write("Loading village boundaries...")

            villages, district_col, taluk_col = load_villages(
                village_path
            )

            progress.write("Calculating village rainfall...")

            villages = calculate_village_rainfall(
                villages,
                interpolator
            )

            warning_pdf = (
                temp_path /
                f"TamilNadu_Rainfall_Warning_{forecast_days}Day.pdf"
            )

            district_pdf = (
                temp_path /
                f"TamilNadu_District_Rainfall_{forecast_days}Day.pdf"
            )

            progress.write("Creating rainfall warning PDF...")

            create_warning_pdf(
                villages,
                district_col,
                taluk_col,
                model_datetime,
                forecast_days,
                warning_pdf
            )

            progress.write("Creating district rainfall PDF...")

            create_district_pdf(
                villages,
                district_col,
                model_datetime,
                forecast_days,
                district_pdf
            )

            progress.update(
                label="Completed successfully!",
                state="complete"
            )

            st.success(
                f"{forecast_days}-day / "
                f"{forecast_days * 24}-hour rainfall report is ready."
            )

            st.subheader("📥 Download")

            col1, col2 = st.columns(2)

            with col1:
                with open(warning_pdf, "rb") as f:
                    st.download_button(
                        "📄 Warning PDF",
                        data=f.read(),
                        file_name=warning_pdf.name,
                        mime="application/pdf",
                        use_container_width=True
                    )

            with col2:
                with open(district_pdf, "rb") as f:
                    st.download_button(
                        "📊 District PDF",
                        data=f.read(),
                        file_name=district_pdf.name,
                        mime="application/pdf",
                        use_container_width=True
                    )

            with st.expander("Forecast information"):
                st.write(
                    f"**Forecast:** {forecast_days} days "
                    f"({forecast_days * 24} hours)"
                )
                st.write(
                    f"**Minimum rainfall:** "
                    f"{float(np.min(rainfall)):.2f} mm"
                )
                st.write(
                    f"**Maximum rainfall:** "
                    f"{float(np.max(rainfall)):.2f} mm"
                )
                st.write(
                    f"**Average grid rainfall:** "
                    f"{float(np.mean(rainfall)):.2f} mm"
                )

        except Exception as e:
            progress.update(
                label="Processing failed",
                state="error"
            )
            st.error(f"Error: {e}")
            st.exception(e)

st.divider()
st.caption(
    "Rainfall values are ECMWF IFS model guidance, not observed rainfall."
)

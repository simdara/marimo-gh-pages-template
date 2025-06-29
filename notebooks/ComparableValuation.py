import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import polars as pl
    import numpy as np
    import altair as alt
    from sklearn.linear_model import LinearRegression
    import openpyxl
    return LinearRegression, alt, mo, np, pd, pl


@app.cell
def _(pd):
    # Read raw excel file
    DataRaw = pd.read_excel("https://simdara.github.io/Data/Data_ASEANFirms.xlsx",dtype=str)
    #DataRaw = pl.from_pandas(DataRaw)
    Data = DataRaw.iloc[:,:]
    return Data, DataRaw


@app.cell
def _(Data, DataRaw, pd):
    # Convert string to float from column 8 onwards
    allColnames = DataRaw.columns # all column names
    cols_to_convert = allColnames[8:]
    #
    for c in cols_to_convert:
        Data[c] = pd.to_numeric(DataRaw[c],errors='coerce')
    return


@app.cell
def _(Data, np):
    #Get unique value of primary industry and industry
    PrimaryIndustry = np.sort(Data["Primary Industry"].dropna().unique())
    Industry = np.sort(Data["Industry"].dropna().unique())
    return (Industry,)


@app.cell
def _(Data, Industry):
    # create dictionary with industry as key and sub-industry as value
    IndustryDict ={}
    for ind in Industry:
        IndustryDict [ind] = Data["Primary Industry"][Data["Industry"]==ind].unique()
    return (IndustryDict,)


@app.cell
def _(IndustryDict, mo):
    # Create downdown ui for the industry
    Indus_dd = mo.ui.dropdown(options=IndustryDict,
                              label="Select Industry", 
                              value="Consumer Finance",searchable=True)
    return (Indus_dd,)


@app.cell
def _(Indus_dd):
    Indus_dd # Industry dropdown ui
    return


@app.cell
def _(Indus_dd, mo):
    # create checkboxes of subindustries once an industry is selected
    checkboxes = mo.ui.array([mo.ui.checkbox(label=subin, value=True) for subin in Indus_dd.value])
    return (checkboxes,)


@app.cell
def _(mo):
    mo.md(r"""**Choose one or more sub-industries**""")
    return


@app.cell
def _(checkboxes, mo):
    mo.vstack(checkboxes)
    return


@app.cell
def _(Data, Indus_dd, checkboxes, pd):
    subInd = pd.DataFrame({"SubInd": Indus_dd.value, "Selected":checkboxes.value}) # convert pd Dataframe
    selectedSubIndustry = subInd["SubInd"][subInd["Selected"]==True].to_numpy()
    usedData = Data[Data["Primary Industry"].isin(selectedSubIndustry)]

    return (usedData,)


@app.cell
def _(mo):
    keywords_ui = mo.ui.text(placeholder="Search in business description....", label="Keywords:", full_width=True)
    keywords_ui
    return (keywords_ui,)


@app.cell
def _(keywords_ui, np, usedData):
    keywords = keywords_ui.value
    kws = keywords.split(";")    
    #
    if keywords=="":
        matchedKW = [True for elem in usedData["Business Description"]]
    else:
        matchedKW = [sum([elem.find(kw)>0 for kw in kws])>0 for elem in usedData["Business Description"]]
    #
    usedData["matchedKW"] = np.array(matchedKW)
    filteredData = usedData[usedData["matchedKW"]].iloc[:,0:-1]
    return (filteredData,)


@app.cell
def _(mo):
    prob_z_score_L = {"1.0%":-2.3263,"2.5%":-1.96, "5.0%": -1.645, "10%": -1.28}
    prob_z_score_U = {"99.0%":2.3263,"97.5%":1.96, "95.0%": 1.645, "90%": 1.28}
    L_dd = mo.ui.dropdown(options=prob_z_score_L, label="Lower Bound", value="2.5%")
    U_dd = mo.ui.dropdown(options=prob_z_score_U, label="Upper Bound", value="97.5%")
    return L_dd, U_dd


@app.cell
def _(L_dd, U_dd, mo):
    mo.vstack([mo.md("**Set outlier:** "),L_dd, U_dd, mo.md("***")])
    return


@app.cell
def _(L_dd, U_dd, filteredData):
    # Find outliers of filtered data from column 8 onwards
    LBound = L_dd.value
    UBound = U_dd.value
    #print(LBound)
    #print(UBound)
    #
    nrows = filteredData.shape[0]
    std =filteredData.iloc[:,8:].std()
    mu = filteredData.iloc[:,8:].mean()
    normFilteredData = filteredData.iloc[:,8:].copy()
    for cn in filteredData.iloc[:,8:].columns:
        normFilteredData[cn] = (filteredData[cn]-mu[cn])/std[cn]

    Ex_Outliers = (normFilteredData>LBound) & (normFilteredData<UBound)
    return (Ex_Outliers,)


@app.cell
def _(mo):
    # Target company inputs
    # Target company data for plot
    requiredInputs = ["Net Income (US$M)", "Total Equity(US$M)", "ROE (%)"]
    optionalInputs = ["Company Name", "Asset Rank"]
    #
    reqInputs_mo = mo.ui.array([mo.ui.text(placeholder="required...", label=input) for input in requiredInputs])
    optInputs_mo = mo.ui.array([mo.ui.text(placeholder="optional...", label=input) for input in optionalInputs])
    #
    mo.vstack([mo.md("**Provide Target Company Info:**")
        ,mo.hstack([mo.vstack(reqInputs_mo), mo.vstack(optInputs_mo)])])
    return optInputs_mo, reqInputs_mo


@app.cell
def _(np, optInputs_mo, pl, reqInputs_mo):
    # collect inputs
    if optInputs_mo.value[0]=="":
        comName = "Target"
    else:
        comName = optInputs_mo.value[0]
    #
    if optInputs_mo.value[1]=="":
        assetRank = np.nan
    else:
        assetRank = int(optInputs_mo.value[1])
    #
    reqInputs = [float(i) if i!="" else np.nan for i in reqInputs_mo.value]
    #
    target_=pl.DataFrame({'Company Name':comName, 'Net Income':reqInputs[0], 'Total Equity, LTM': reqInputs[1], 'ROE, LTM':reqInputs[2], 'Asset Rank': assetRank, "Peer_Target": "Target"})
    target_
    return (target_,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ***
    ###**Histogram and descriptive statitics**
    """
    )
    return


@app.cell
def _(Indus_dd, filteredData, mo):
    # Create downdown ui for columns
    if Indus_dd.selected_key in ["Banks", "Consumer Finance", "Financial Services", "Insurance"]:
        default_metric = "P/BV, LTM"
    else:
        default_metric = "TEV/EBITDA, LTM"

    Metrics_dd = mo.ui.dropdown(options=filteredData.columns[8:],
                              label="Select Key Metrics 1", 
                              value=default_metric)
    Metrics_dd2 = mo.ui.dropdown(options=filteredData.columns[8:],
                              label="Select Key Metrics 2", 
                              value="P/E, Normalized, LTM")

    mo.hstack([Metrics_dd, Metrics_dd2])
    return Metrics_dd, Metrics_dd2


@app.cell
def _(Ex_Outliers, Hist_keymetrics, Metrics_dd, filteredData, mo, pd):
    selMetric = filteredData[["No",Metrics_dd.value]]
    selMetric_exOut = selMetric[Ex_Outliers[Metrics_dd.value]==True] # excluded outliers
    #
    Chart_1 = Hist_keymetrics(selMetric_exOut)
    #
    desc1_pd = pd.DataFrame(selMetric_exOut.iloc[:, 1].describe())
    # use statistic as index and transpose
    desc1_pd = desc1_pd.T
    # change number format
    for col in desc1_pd.columns:
        desc1_pd[col] = desc1_pd[col].apply(lambda x: f"{x:.2f}")

    table1 = mo.md(desc1_pd.to_markdown(index=False))
    return Chart_1, table1


@app.cell
def _(Ex_Outliers, Hist_keymetrics, Metrics_dd2, filteredData, mo, pd):
    selMetric2 = filteredData[["No",Metrics_dd2.value]]
    selMetric2_exOut = selMetric2[Ex_Outliers[Metrics_dd2.value]==True] # excluded outliers
    #
    Chart_2 = Hist_keymetrics(selMetric2_exOut)
    #
    desc2_pd = pd.DataFrame(selMetric2_exOut.iloc[:, 1].describe())
    # use statistic as index and transpose
    desc2_pd = desc2_pd.T
    # change number format
    for col2 in desc2_pd.columns:
        desc2_pd[col2] = desc2_pd[col2].apply(lambda x: f"{x:.2f}")

    table2 = mo.md(desc2_pd.to_markdown(index=False))
    return Chart_2, table2


@app.cell
def _(alt):
    def Hist_keymetrics(keyMetric):
        hisChart = (
            alt.Chart(keyMetric)
            .mark_bar()
            .encode(
                x=alt.X(keyMetric.columns[1], type="quantitative", bin=True, title=keyMetric.columns[1]),
                y=alt.Y("count()", type="quantitative", title="Number of records"),
                tooltip=[
                    alt.Tooltip(
                        keyMetric.columns[1],
                        type="quantitative",
                        bin=True,
                        title=keyMetric.columns[1],
                        format=",.2f",
                    ),
                    alt.Tooltip(
                        "count()",
                        type="quantitative",
                        format=",.0f",
                        title="Number of records",
                    ),
                ],
            ).properties(width="container",height=300).configure_view(stroke=None)
        )
        return hisChart

    return (Hist_keymetrics,)


@app.cell
def _(Chart_1, Chart_2, mo, table1, table2):
    mo.hstack([mo.vstack([Chart_1,table1]),mo.vstack([Chart_2,table2])])
    return


@app.cell
def _(Ex_Outliers, filteredData):
    # Treated outliers by setting them none
    treatedData = filteredData.copy()
    for j in Ex_Outliers.columns:
        treatedData.loc[~Ex_Outliers[j],j]=None
    return (treatedData,)


@app.cell
def _(LinearRegression, alt, np, pl):
    def scatterplot(peerData,TargetData, Xname, Yname):
        # Regression
        model = LinearRegression()
        xy = peerData[["Company Name",Xname, Yname]].dropna() # drop nulls
        model.fit(xy[[Xname]], xy[Yname])

        R2 = model.score(xy[[Xname]], xy[Yname])    
        ##--------------
        eqtext = f'y = {model.coef_[0]:.2f} x + {model.intercept_:.2f};' +f'\n(R2 = {R2:.2f})'

        scatter = alt.Chart(xy).mark_point(filled=True,size =50.0).encode(
                x=alt.X(field=Xname, type='quantitative'),
                y=alt.Y(field=Yname, type='quantitative'),
                #color='Peer_Target',
                tooltip=[
                    alt.Tooltip(field="Company Name"),
                    alt.Tooltip(field=Xname, format=',.2f'),
                    alt.Tooltip(field=Yname, format=',.2f')
                ]
            ).properties(
                title=Xname + " vs " + Yname + " Scatter Plot with Regression Line",
                height=450,
                width='container'
        )

        # Add equation line and text
        reg_line = scatter.transform_regression(
            Xname, Yname,
            method="linear",
        ).mark_line()

        scatterText = reg_line.mark_text(
            align='left',
            baseline='top',
            fontSize=13,
            dx=10,
            dy=10,
            text=eqtext,
            color = 'cyan'
        ).encode(
            x=alt.value(10),  # Position text
            y=alt.value(10)
        )
        ##---------
        if (TargetData[Xname]!=np.nan)[0]:

            # Target company data for plot
            TData_extended = TargetData.with_columns([pl.lit(model.predict(TargetData[[Xname]])).alias(Yname)])
            # add label
            textLabel =f"{TData_extended['Company Name'][0]} ({TData_extended[Xname][0]:.2f}, {TData_extended[Yname][0]:.2f})"
            TData_extended = TData_extended.with_columns([pl.lit(textLabel).alias("label")])
            #-----------------------------------------------------------------
            scatter_tar = alt.Chart(TData_extended).mark_point(filled=True,size =100.0,color="red").encode(
                    x=alt.X(field=Xname, type='quantitative'),
                    y=alt.Y(field=Yname, type='quantitative'),
                    #color='Peer_Target',
                    tooltip=[
                        alt.Tooltip(field="Company Name"),
                        alt.Tooltip(field=Xname, format=',.2f'),
                        alt.Tooltip(field=Yname, format=',.2f')
                    ]
                ).properties(
                    title = Xname + " vs " + Yname + " Scatter Plot with Regression Line",
                    height = 450,
                    width='container',
                )
            scatter_tar_text = alt.Chart(TData_extended).mark_text(
                dx = -15, 
                dy = -15,
                color = "red",
                fontSize= 15,
                ).encode(
                    x=alt.X(field=Xname, type='quantitative'),
                    y=alt.Y(field=Yname, type='quantitative'),
                    text = 'label'
            )
            chart = scatter + scatterText + reg_line + scatter_tar + scatter_tar_text
        else:
            chart = scatter + scatterText + reg_line
        return chart
    return (scatterplot,)


@app.cell
def _(mo, scatterplot, target_, treatedData):
    target=target_[["Company Name", "ROE, LTM", "Peer_Target"]]
    chart3 = scatterplot(treatedData, target,"ROE, LTM", "P/BV, LTM")

    subtitle_div = mo.md("###**ScatterPlot with Regression Line**")
    scatterplotdiv = mo.accordion({'Scatter Plot with regression👉': chart3})
    mo.vstack([subtitle_div, scatterplotdiv])
    return


@app.cell
def _(mo, treatedData):
    sub_div2 = mo.md("###**Table of ASEAN peers**")
    tablediv = mo.accordion({"Table of matched ASEAN firms👉": treatedData})

    mo.vstack([sub_div2,tablediv])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

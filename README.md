# Business_Need
Employed by a wealthy investor that specialises in purchasing assets that are undervalued.  This investor’s due diligence on all purchases includes a detailed analysis of the data that underlies the business, to try to understand the fundamentals of the business and especially to identify opportunities to drive profitability by changing the focus of which products or services are being offered.

The investor is interested in purchasing TellCo, an existing mobile service provider in the Republic of Pefkakia.  TellCo’s current owners have been willing to share their financial information but have never employed anyone to look at the data that is generated automatically by their systems.

As an employer wants the investor wants a report to analyse opportunities for growth and make a recommendation on whether TellCo is worth buying or selling.  This done by analysing a telecommunication dataset that contains useful information about the customers & their activities on the network. Then deliver insights to the employer through an easy-to-use web-based dashboard and a written report. 

## Task 1 - User Overview Analysis: 
For the actual telecom dataset, you‘re expected to conduct a full User Overview analysis & the following sub-tasks are your guidance: 
- Start by identifying the top 10 handsets used by the customers.
- Then, identify the top 3 handset manufacturers
- Next, identify the top 5 handsets per top 3 handset manufacturer
- Make a short interpretation and recommendation to marketing teams

 
**Task 1.1** - The employer wants to have an overview of the users’ behavior on those applications.   
Aggregate per user the following information in the column  
- Number of xDR sessions
- Session duration
- The total download (DL) and upload (UL) data
- The total data volume (in Bytes) during this session for each application


**Task 1.2** - Conduct an exploratory data analysis on those data & communicate useful insights. Ensure that identifying and treating all missing values and outliers in the dataset by replacing them with the mean or any possible solution of the corresponding column.
- Report about the following using Python script and slide  :
- Describe all relevant variables and associated data types (slide). findings. 
- Variable transformations – segment the users into the top five decile classes based on the total duration for all sessions and compute the total data (DL+UL) per decile class. 
- Analyze the basic metrics (mean, median, etc) in the Dataset (explain) & their importance for the global objective.
- Conduct a Non-Graphical Univariate Analysis by computing dispersion parameters for each quantitative variable and provide useful interpretation. 
- Conduct a Graphical Univariate Analysis by identifying the most suitable plotting options for each variable and interpret your findings.
- Bivariate Analysis – explore the relationship between each application & the total DL+UL data using appropriate methods and interpret your result
- Correlation Analysis – compute a correlation matrix for the following variables and interpret your findings: Social Media data, Google data, Email data, YouTube data, Netflix data, Gaming data, and Other data 
- Dimensionality Reduction – perform a principal component analysis to reduce the dimensions of your data and provide a useful interpretation of the results (Provide your interpretation in four (4) bullet points maximum). 


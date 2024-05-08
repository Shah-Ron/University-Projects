# Library to read the Downloaded Excel file
library(readxl)
library(writexl)
library(tidyr)

# Getting the current working directory to set the path
path <- getwd()

# Getting the filename of the downloaded excel file from the text named filename.txt
filename <- paste0(path,"/filename.txt")
filename <- readLines(filename)

# The path to the excel file
path <- paste0(path,"/",filename)

# Since the excel file consist of different Sheets, getting all the sheet names
sheet_names <- excel_sheets(path)

# Initializing for the List of Data frame
all_data <- list()

for (sheet_name in sheet_names){
  
  # Reading the data in individual sheets
  data <- as.data.frame(read_excel(path, sheet = sheet_name))
  
  # Observed that the date format in the data was different, so changed it to "YYYY-MM-DD"
  
  
  # Assigning the data in the list
  all_data[[sheet_name]] <- data
}
rm(data)

cleaning_df <- function(df) {
  for (i in 1:length(df)) {
    
    # Add column names as the first row
    df[[i]] <- rbind(as.character(names(df[[i]])), df[[i]])
    
    # Remove column names
    colnames(df[[i]]) <- NULL
    
    # Transpose dataframe
    df[[i]] <- as.data.frame(t(df[[i]]))
    
    # Set "Date" in the first row
    
    df[[i]][1, 1] <- "Date"
    
    # Set column names as the first row
    colnames(df[[i]]) <- df[[i]][1, ]
    
    # Remove the first row
    df[[i]] <- df[[i]][-1, ]
    
    # Convert Dates into Date format
    df[[i]]$Date <- as.Date(as.numeric(df[[i]]$Date), origin = "1899-12-30")

    
    # Replace NA values with 0
    df[[i]][is.na(df[[i]])] <- 0
    
  }
  return(df)
}

delete_repetition_df <- function(df1, df2, start_i, end_i){
  
  for (i in start_i:end_i) {
    
    # Repetitive count
    addon <- i*8
    
    # Getting the value for Locations columns
    location_value <- colnames(df2)[(2+addon)]
    
    # Selecting the columns for binding
    selected_df <- df2[, c(1,(3+addon):(9+addon))]
    
    # Adding location column to the data set
    selected_df$Location <- location_value
    
    # Binding the data
    df1 <- rbind(df1,selected_df)
  }
  
  return(df1)
}

make_csv <- function(df, filenames){
  
  for (i in 1:length(filenames)) {
    write.csv(df[[i]], filenames[i], row.names = FALSE)
  }
  return(df)
}

pivot_data <- function(df,name_to,end_col){
  for(i in 1:length(df)){
    col_end <- as.numeric(end_col[i])
    df[[i]] <- pivot_longer(df[[i]],
                          cols = 2: col_end,
                          names_to = name_to[i],
                          values_to = "Observations"
                          )
  }
  return(df)
}

# Dealing with the first sheet which is Population

Population <- all_data[["Population"]]

# Seperate the Population Sheet into Total Prisoners, Male Prisoners and Women Prisoners

list_population_df <- list()
list_population_df[[1]] <- Population[18:137, 2: ncol(Population)]  #Male Prisoners Data  
list_population_df[[2]] <- Population[147:170, 2: ncol(Population)] #Women Prisoners Data

# Transposing the data to convert dates as rows and providing a cleaner look. 
# General data cleaning date format and percentage values are changed back to out of 100.

list_population_df <- cleaning_df(list_population_df)

filenames <- c( 
  "Prisoner Population.csv", 
  "Age Group.csv",
  "Ethnicity.csv",
  "Security Class.csv", 
  "Offence Type.csv"
  )

# Total Prisoners Data is cleaned by the general cleaning process
# Male Prisoners Data Cleaning

male_prisoner_df <- list_population_df[[1]][,1:9]
male_prisoner_df$Location <- colnames(male_prisoner_df)[2]
male_prisoner_df <- male_prisoner_df[,-2]

list_population_df[[1]] <- delete_repetition_df(male_prisoner_df,list_population_df[[1]],1,14)
rm(male_prisoner_df)

# Women Prisoners Data Cleaning

women_prisoner_df <- list_population_df[[2]][,1:9]
women_prisoner_df$Location <- colnames(women_prisoner_df)[2]
women_prisoner_df <- women_prisoner_df[,-2]

list_population_df[[2]] <- delete_repetition_df(women_prisoner_df,list_population_df[[2]],1,2)

rm(women_prisoner_df)

col_end <- list()
column_names <- c("Population Type", "Population Type")
for (i in 1:length(column_names)) {
  col_end[i] <- ncol(list_population_df[[i]]) - 1
}


list_population_df <- pivot_data(list_population_df, column_names,col_end)
list_population_df[[1]]$'Gender' <- "Male"
list_population_df[[2]]$'Gender' <- "Female"

all_data[[1]] <- rbind(list_population_df[[1]],list_population_df[[2]])

all_data[[1]]$Observations <- as.numeric(all_data[[1]]$Observations)

all_data[[1]]$Observations[all_data[[1]]$`Population Type` == "Population percentage"] <- 
  all_data[[1]]$Observations[all_data[[1]]$`Population Type` == "Population percentage"] * 100

rm(Population)

# Dealing with the next four sheets in the data set, which are Age, Ethnicity, Security class and Offence Type, together because they are similar

list_AESO_df <- list() # AES stands for Age, Ethnicity, Security Class and Offence Type 
list_AESO_df[[1]] <- all_data[["Age"]]
list_AESO_df[[2]] <- all_data[["Ethnicity"]]
list_AESO_df[[3]] <- all_data[["Security class"]]
list_AESO_df[[4]] <- all_data[["Offence type"]]

list_AESO_df <- cleaning_df(list_AESO_df)

list_AESO_df[[4]] <- list_AESO_df[[4]][,1:24]

column_names <- c("Age group", "Ethnicity", "Security class", "Offense type")
for(i in 1:length(list_AESO_df)){
  col_end[i] <- ncol(list_AESO_df[[i]])
}

list_AESO_df <- pivot_data(list_AESO_df, column_names, col_end)

for(i in 1:4){
  all_data[[i+1]] <- list_AESO_df[[i]]
  all_data[[i+1]]$Observations <- as.numeric(all_data[[i+1]]$Observations) * 100
}

rm(col_end)
rm(list_AESO_df)
rm(list_population_df)

all_data <- make_csv(all_data, filenames)
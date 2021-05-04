########################
##IST 707: Final Project
##Visualization for some attribtues
##Due Date: 04/28/2020
########################

# Library Packages
library(dplyr)
library(RColorBrewer)
library(ggplot2)
library(highcharter)

# Load the clean dataset
df <- read.csv("C:/Users/Linxi Feng/Desktop/2020SP/IST707/Final Project/Final Project Report/airbnb-listings-pre-clean.csv"
               , header = TRUE
               , stringsAsFactors = FALSE)

# Inspect the dataset
str(df)
summary(df)
View(df)

# Explore the dataset

# ---- Property type ---- #
pro <- table(df$Property_Type) %>% as.data.frame()
colnames(pro) <- c("Property type", "Number")
View(pro)

coul <- brewer.pal(8, "Set2") 
pro %>%
  ggplot(aes(x=reorder(pro$`Property type`,-pro$Number), y=pro$Number, fill=pro$`Property type`)) +
  geom_bar(stat="identity") +
  theme(
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position="none"
  ) +
  ggtitle("The distributin of Property Type") +
  xlab("") +
  ylab("Number") +
  theme(axis.text.x = element_text(angle = 90)) +
  theme(panel.background = element_blank())


# ---- Room type ---- #
room <- df %>% count(Room_Type) %>% 
  mutate(percentage = round((n/sum(n))*100)) %>%
  arrange(desc(percentage)) %>% 
  as.data.frame()

hciconarray(room$Room_Type, room$percentage, size = 5) %>%
  hc_title(text="Proportion of Room Type")



# ---- Accommodates ---- #
acc <- table(df$Accommodates) %>% as.data.frame()
colnames(acc) <- c("Accommadate number", "Frequency")
View(acc)

acc %>%
  ggplot(aes(x=acc$`Accommadate number`, y=acc$Frequency, fill=acc$`Accommadate number`)) +
  geom_bar(stat="identity") +
  theme(
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position="none"
  ) +
  ggtitle("How many guest can they accomodate?") +
  xlab("") +
  ylab("Frequency") +
  theme(panel.background = element_blank())





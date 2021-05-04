################################## 
##### IST 707 Final Project
##### Zequn Che & Jiyu Feng
##### Airbnb Price Analysis
##### Due Time: 04/28/2020
##################################


# library packages
library(dplyr)
library(arules)


# load the dataset
df <- read.csv(choose.files(), header = TRUE, stringsAsFactors = FALSE)
View(df)

# inspect the data
str(df)
summary(df)
dim(df)

colnames(df)[1] <-"Host.Total.Listings.Count"

# convert data types: numeric variables
df1 <- df %>% 
  mutate(Host.Total.Listings.Count = as.numeric(Host.Total.Listings.Count),
         Neighbourhood.Cleansed = as.factor(Neighbourhood.Cleansed),
         Property.Type = as.factor(Property.Type),
         Room.Type = as.factor(Room.Type),
         Accommodates = as.numeric(Accommodates),
         Bathrooms = as.numeric(Bathrooms),
         Bedrooms = as.numeric(Bedrooms),
         Beds = as.numeric(Beds),
         Bed.Type = as.factor(Bed.Type),
         Price = as.numeric(Price),
         Cleaning.Fee = as.numeric(Cleaning.Fee),
         Cancellation.Policy = as.factor(Cancellation.Policy))%>%
select(c("Host.Total.Listings.Count","Neighbourhood.Cleansed",
         "Property.Type","Room.Type","Accommodates","Bathrooms",
         "Bedrooms","Beds","Bed.Type","Price",
         "Cleaning.Fee","Cancellation.Policy"))

# Check missing values
sapply(df1, function(x)sum(is.na(x)))
df1 <- na.omit(df1) 
sapply(df1, function(x)sum(is.na(x))) # double check NAs

# convert chr into factor
#df_clean <- df1 %>%
#  mutate_if(is.character, as.factor)

# set the level of the list price
df_clean <- df1 %>%
  mutate(Price = cut(Price, breaks = c(0,181,Inf)
                     , labels = c("low", "high")))

# Inspect the new dataset
str(df_clean)
summary(df_clean)
View(df_clean)




############# Association Rule ################
library(arulesViz)

df_clean_association <- df_clean %>% mutate_if(is.numeric,as.factor)

df_2 <-discretizeDF(df_clean_association, methods = NULL, default = NULL)

rules_1 <- df_2 %>% apriori(parameter = list(support = 0.015, confidence = 0.78, maxlen = 5)
                            ,appearance = list(rhs = c("Price=low", "Price=high"))
                            , control = list (verbose=F))

rules_1 <- rules_1[!is.redundant(rules_1)]

arules::inspect(rules_1)

df_rule = data.frame(
  lhs = labels(lhs(rules_1)),
  rhs = labels(rhs(rules_1)), 
  rules_1@quality)

df_rule <- df_rule %>% arrange(-lift)
View(df_rule)

df_rule_low <- df_rule %>% filter(df_rule$rhs == "{Price=low}")
View(df_rule_low)

df_rule_high <- df_rule %>% filter(df_rule$rhs == "{Price=high}")
View(df_rule_high)

#########################################################################################################
# Decision Tree
library(C50)
#install.packages('gmodels')
library(gmodels)
library(partykit)
library(RColorBrewer)
library(dplyr)
library(rpart)
#install.packages('rpart.plot')
library(rpart.plot)
df_DT <- df_clean %>% select(-c("Neighbourhood.Cleansed"))

##################################
# Rpart
# Adjusting the Complexity Parameter 
set.seed(7)
tree <- rpart(Price ~ ., 
              data = df_DT, 
              control = rpart.control(cp = 0.001,minsplit = 10,minbucket = 10))
prp(tree, 
    faclen = 0, 
    cex = 0.8, 
    extra = 1)  

cred_conf_matrix <- table(df_DT$Price,
                          predict(tree,
                                  type="class"))

cred_conf_matrix


# Pruning the Decision Tree
printcp(tree)

min_xerror <- tree$cptable[,"xerror"] %>% 
  which.min()

bestcp <- tree$cptable[min_xerror,"CP"]


# Step 3: Prune the tree using the best cp.
tree.pruned <- prune(tree, cp = bestcp)

prp(tree.pruned, 
    faclen = 0, 
    cex = 0.8, 
    extra = 1)  

cred_conf_matrix_2 <- table(df_DT$Price,
                            predict(tree.pruned,
                                    type="class"))
cred_conf_matrix_2

# Clarifying the row and column names
rownames(cred_conf_matrix) <- paste("Actual", rownames(cred_conf_matrix), sep = ":")
colnames(cred_conf_matrix) <- paste("Pred", colnames(cred_conf_matrix), sep = ":")

rownames(cred_conf_matrix_2) <- paste("Actual", rownames(cred_conf_matrix_2), sep = ":")
colnames(cred_conf_matrix_2) <- paste("Pred", colnames(cred_conf_matrix_2), sep = ":")


# Pruned vs. Unpruned Tree Confusion Matrices
print(cred_conf_matrix)
print(cred_conf_matrix_2)      


# Pruned vs. Unpruned Trees Plotted
prp(tree, faclen = 0, cex = 0.8, extra = 1)  
prp(tree.pruned, faclen = 0, cex = 0.8, extra = 1)  

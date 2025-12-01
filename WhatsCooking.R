# load packages

library(tidyverse)
library(tidymodels)
library(jsonlite)
library(tidytext)

# read in data

train <- read_file('train.json') %>%
  fromJSON()
test <- read_file('test.json') %>%
  fromJSON()

# tokenize

train <- train %>%
  unnest(ingredients)
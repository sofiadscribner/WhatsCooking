# load packages

library(tidyverse)
library(tidymodels)
library(jsonlite)
library(tidytext)
library(stringr)

# read in data

train <- read_file('train.json') %>%
  fromJSON()%>%
  mutate(cuisine = factor(cuisine))
test <- read_file('test.json') %>%
  fromJSON()

# tokenize

train <- train %>%
  unnest(ingredients)

test <- test %>%
  unnest(ingredients)

# create number of ingredients feature 

ingredient_counts <- train %>%
  count(id, name = "n_ingredients")

ingredient_counts_test <- test %>%
  count(id, name = "n_ingredients")

train <- train %>%
  left_join(ingredient_counts, by = "id")

test <- test %>%
  left_join(ingredient_counts_test, by = "id")

# create has_beef feature

train <- train %>%
  mutate(
    contains_beef = if_else(
      str_detect(ingredients, regex("beef", ignore_case = TRUE)),
      1, 0
    )
  )

test <- test %>%
  mutate(
    contains_beef = if_else(
      str_detect(ingredients, regex("beef", ignore_case = TRUE)),
      1, 0
    )
  )

# create average ingredient character length

avg_length <- train %>%
  mutate(ing_char_len = nchar(ingredients)) %>%
  group_by(id) %>%
  summarise(
    avg_ing_char_len = mean(ing_char_len)
  )

avg_length_test <- test %>%
  mutate(ing_char_len = nchar(ingredients)) %>%
  group_by(id) %>%
  summarise(
    avg_ing_char_len = mean(ing_char_len)
  )


train <- train %>%
  left_join(avg_length, by = "id")

test <- test %>%
  left_join(avg_length_test, by = "id")


# finalize as one row per id

train <-  train %>%
  group_by(id, cuisine) %>% 
  summarise(
    n_ingredients = n(),
    contains_beef = max(contains_beef),
    avg_ing_char_len = mean(nchar(ingredients)),
    .groups = "drop"
  )


test <-  test %>%
  group_by(id) %>% 
  summarise(
    n_ingredients = n(),
    contains_beef = max(contains_beef),
    avg_ing_char_len = mean(nchar(ingredients)),
    .groups = "drop"
  )


# fit a random forest

# recipe

rf_recipe <- recipe(cuisine ~ n_ingredients + contains_beef + avg_ing_char_len,
                    data = train)

# model

rf_model <- rand_forest(
  trees = 1000,
  mtry = 3,
  min_n = 5
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# workflow

rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(rf_recipe)

# fit model

rf_fit <- rf_workflow %>% fit(data = train)

# make preds

rf_predictions <- predict(rf_fit, test) %>%
  bind_cols(test)

# format for sub

submission <- test %>%
  select(id) %>%
  bind_cols(rf_predictions %>% select(cuisine = .pred_class))

# save

write_csv(submission, "submission.csv")


# try using tfidf


# load packages

library(tidyverse)
library(tidymodels)
library(jsonlite)
library(tidytext)
library(stringr)
library(textrecipes)

# read in data

train <- read_file('train.json') %>%
  fromJSON()%>%
  mutate(cuisine = factor(cuisine))
test <- read_file('test.json') %>%
  fromJSON()

# define recipe with tfidf

tfidf_recipe <- recipe(cuisine ~ ingredients,
                    data = train) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens=500) %>%
  step_tfidf(ingredients)
  
# model

rf_model <- rand_forest(
  trees = 1000,
  mtry = 3,
  min_n = 5
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# workflow

tfidf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(tfidf_recipe)

# fit model

rf_fit <- tfidf_workflow %>% fit(data = train)

# make preds

rf_predictions <- predict(rf_fit, test) %>%
  bind_cols(test)

# format for sub

submission <- test %>%
  select(id) %>%
  bind_cols(rf_predictions %>% select(cuisine = .pred_class))

# save

write_csv(submission, "submission.csv")



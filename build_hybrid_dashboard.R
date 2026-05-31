library(data.table)
library(dplyr)
library(lightgbm)

# ------------------------------------------------
# Paths
# ------------------------------------------------

data_path <- "C:/Users/User/Documents/pitches_2024_full.csv"
model_path <- "C:/Users/User/Documents/deployment/lightgbm_strike_model.txt"
bayes_v3_path <- "C:/Users/User/Documents/deployment/bayesian_dashboard_grid_v3.csv"
bayes15_path <- "C:/Users/User/Documents/deployment/bayesian_dashboard_grid_15.csv"
output_path <- "C:/Users/User/Documents/deployment/hybrid_dashboard_grid_portfolio_small.csv"

selected_pitch_types <- c("FF", "SI", "SL", "CH", "CU")
grid_size <- 15

logit <- function(p) log(p / (1 - p))
inv_logit <- function(x) 1 / (1 + exp(-x))
eps <- 1e-6

# ------------------------------------------------
# 1. Build Bayesian 15x15 grid from v3
# ------------------------------------------------

cat("Reading Bayesian v3 grid...\n")

bayes_grid <- fread(
  bayes_v3_path,
  select = c(
    "plate_x", "plate_z",
    "pitch_type", "balls", "strikes",
    "stand", "p_throws", "catcher",
    "catcher_name",
    "bayes_prob", "baseline_prob"
  )
)

target_x <- seq(-2, 2, length.out = grid_size)
target_z <- seq(0, 5, length.out = grid_size)

bayes_grid[, ix := pmin(pmax(round((plate_x - min(target_x)) / (max(target_x) - min(target_x)) * (grid_size - 1)) + 1, 1), grid_size)]
bayes_grid[, iz := pmin(pmax(round((plate_z - min(target_z)) / (max(target_z) - min(target_z)) * (grid_size - 1)) + 1, 1), grid_size)]

bayes_grid[, plate_x := target_x[ix]]
bayes_grid[, plate_z := target_z[iz]]

bayes15 <- bayes_grid[
  ,
  .(
    bayes_prob = mean(bayes_prob, na.rm = TRUE),
    baseline_prob = mean(baseline_prob, na.rm = TRUE),
    catcher_name = first(catcher_name)
  ),
  by = .(
    plate_x, plate_z,
    pitch_type, balls, strikes,
    stand, p_throws, catcher
  )
]

bayes15[, bayesian_logit_effect :=
          logit(pmin(pmax(bayes_prob, eps), 1 - eps)) -
          logit(pmin(pmax(baseline_prob, eps), 1 - eps))]

fwrite(bayes15, bayes15_path)

cat("Saved Bayesian 15x15 grid.\n")

rm(bayes_grid)
gc()

# ------------------------------------------------
# 2. Rebuild LightGBM data structure
# ------------------------------------------------

cat("Reading full pitch data...\n")

full_data <- read.csv(data_path)

model_lgb_data <- full_data %>%
  filter(
    pitch_type %in% selected_pitch_types,
    description %in% c("called_strike", "ball")
  ) %>%
  mutate(
    called_strike = ifelse(description == "called_strike", 1, 0),
    pitch_type = factor(pitch_type),
    stand = factor(stand),
    p_throws = factor(p_throws),
    catcher = factor(fielder_2)
  ) %>%
  select(
    called_strike,
    plate_x,
    plate_z,
    pfx_x,
    pfx_z,
    release_speed,
    release_spin_rate,
    balls,
    strikes,
    pitch_type,
    stand,
    p_throws,
    catcher
  ) %>%
  na.omit()

feature_cols <- colnames(
  model.matrix(called_strike ~ . - 1, data = model_lgb_data)
)

lgb_model <- lgb.load(model_path)

top_catchers <- model_lgb_data %>%
  count(catcher, sort = TRUE) %>%
  slice_head(n = 9)

catcher_list <- as.character(
  top_catchers$catcher
)

pitch_type_means <- model_lgb_data %>%
  group_by(pitch_type) %>%
  summarise(
    pfx_x = mean(pfx_x, na.rm = TRUE),
    pfx_z = mean(pfx_z, na.rm = TRUE),
    release_speed = mean(release_speed, na.rm = TRUE),
    release_spin_rate = mean(release_spin_rate, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------
# 3. Build LightGBM dashboard grid
# ------------------------------------------------

grid_df <- expand.grid(
  plate_x = target_x,
  plate_z = target_z,
  pitch_type = selected_pitch_types,
  balls = 0:3,
  strikes = 0:2,
  stand = c("L", "R"),
  p_throws = c("L", "R"),
  catcher = catcher_list,
  KEEP.OUT.ATTRS = FALSE
)

grid_df <- grid_df %>%
  mutate(
    pitch_type = factor(pitch_type, levels = levels(model_lgb_data$pitch_type)),
    stand = factor(stand, levels = levels(model_lgb_data$stand)),
    p_throws = factor(p_throws, levels = levels(model_lgb_data$p_throws)),
    catcher = factor(catcher, levels = levels(model_lgb_data$catcher))
  ) %>%
  left_join(pitch_type_means, by = "pitch_type")

grid_df$lightgbm_prob <- NA_real_

chunk_size <- 50000

for (i in seq(1, nrow(grid_df), by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, nrow(grid_df))
  
  grid_x <- model.matrix(
    ~ . - 1,
    data = grid_df[idx, ] %>%
      select(
        plate_x, plate_z,
        pfx_x, pfx_z,
        release_speed, release_spin_rate,
        balls, strikes,
        pitch_type, stand, p_throws, catcher
      )
  )
  
  missing_cols <- setdiff(feature_cols, colnames(grid_x))
  
  for (col in missing_cols) {
    grid_x <- cbind(grid_x, 0)
    colnames(grid_x)[ncol(grid_x)] <- col
  }
  
  grid_x <- grid_x[, feature_cols]
  
  grid_df$lightgbm_prob[idx] <- predict(lgb_model, grid_x)
  
  cat("Finished LightGBM rows:", max(idx), "of", nrow(grid_df), "\n")
}

# ------------------------------------------------
# 4. Merge Bayesian framing effect
# ------------------------------------------------

grid_dt <- as.data.table(grid_df)
grid_dt[, catcher := as.character(catcher)]

setDT(bayes15)
bayes15[, catcher := as.character(catcher)]

hybrid_grid <- merge(
  grid_dt,
  bayes15[
    ,
    .(
      plate_x, plate_z,
      pitch_type, balls, strikes,
      stand, p_throws, catcher,
      catcher_name,
      bayesian_logit_effect
    )
  ],
  by = c(
    "plate_x", "plate_z",
    "pitch_type", "balls", "strikes",
    "stand", "p_throws", "catcher"
  ),
  all.x = TRUE
)

hybrid_grid[is.na(bayesian_logit_effect), bayesian_logit_effect := 0]
hybrid_grid[is.na(catcher_name), catcher_name := catcher]

hybrid_grid[, adjusted_prob :=
              inv_logit(
                logit(pmin(pmax(lightgbm_prob, eps), 1 - eps)) +
                  bayesian_logit_effect
              )]

hybrid_grid[, framing_effect := adjusted_prob - lightgbm_prob]

fwrite(hybrid_grid, output_path)

cat("Done. Saved hybrid dashboard grid to:\n")
cat(output_path, "\n")
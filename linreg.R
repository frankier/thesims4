data <- read.csv("final.csv")

prep_queue_mod <- lm(prep_queue ~ X0 + X1 + X2 + X3 + X4 + X5, data=data)
surg_backed_up_mod <- lm(surg_backed_up ~ X0 + X1 + X2 + X3 + X4 + X5, data=data)
surg_utilisation_mod <- lm(surg_utilisation ~ X0 + X1 + X2 + X3 + X4 + X5, data=data)

summary(prep_queue_mod)
summary(surg_backed_up_mod)
summary(surg_utilisation_mod)


## 1) Header 

# Script
# Purpose
# Authors
# Date

## 2) Libraries


library(dplyr)
library(ggplot2)
library(readr)




## 3) Global Option



## 4) Data import ##

d.municipalities <- read.csv2("data/official-building-adress-register-switzerland.csv")
d.solarpotential <- read.csv("data/solarpotential-switerzland-roofs-facades.csv")
d.heatpotential <- read.csv("data/heatsupplier-potential.csv")
d.vehicleenergydemand <- read.csv("data/energydemand-pw-bev-phev-municipalities.csv")
d.vehiclefleet <- read.csv("data/vehicle-fleet-pw-municipalities.csv")
d.newvehicleregistration <- read.csv("data/newvehicleregistration-pw-municipalities.csv")
d.vehiclemileage <- read.csv("data/vehiclemileage-pw-municipalities.csv")


## Renaming

d.municipalities.renamed <- d.municipalities %>%
  rename(
    buildingadressid = ADR_EGAID,
    streetid = STR_ESID,
    buildingid = BDG_EGID,
    inputid = ADR_EDID,
    streetname = STN_LABEL,
    housenumber = ADR_NUMBER,
    buildingcategory = BDG_CATEGORY,
    buildingname = BDG_NAME,
    postcode = ZIP_LABEL,
    bfs_municipalitynr = COM_FOSNR,
    bfs_municipalityname = COM_NAME,
    canton = COM_CANTON,
    adressstatus = ADR_STATUS,
    officialadress = ADR_OFFICIAL,
    lastadresschange = ADR_MODIFIED,
    eastcoordinate = ADR_EASTING,
    northcoordinate = ADR_NORTHING, 
  ) %>%  filter(
             bfs_municipalitynr >= 4000 & bfs_municipalitynr <= 4399,
     )


d.solarpotential.renamed <- d.solarpotential %>% 
  rename(
  bfs_municipalitynr = MunicipalityNumber,
  bfs_municipalityname = MunicipalityName,
  canton = Canton,
  country = Country,
  solarpotential.electricity = Scenario4_RoofsFacades_PotentialSolarElectricity_GWh,
  solarpotential.heat = Scenario4_RoofsFacades_PotentialSolarHeat_GWh
) %>%
  # Deletion of not relevant fields
  select(
    -starts_with(c(
      "Scenario1", # Will now be ignored since the column was renamed
      "Scenario2",
      "Scenario3",
      "Fact",
      "Metho",
      "HIST"
    ))
  )





d.heatpotential.renamed <- d.heatpotential %>%
  rename(
    bfs_municipalitynr = xtf_id,
    bfs_municipalityname = Name,
    heatpotential = HeatPotential_MWha,
    heatsupplier = HeatSupplierCategory) %>% 
  select(c(
    "bfs_municipalitynr",
    "bfs_municipalityname",
    "heatpotential",
    "heatsupplier"
  ))


d.vehicleenergydemand.renamed <- d.vehicleenergydemand %>%
  rename(
    bfs_municipalitynr = Region_ID,
    year = Jahr,
    bfs_municipalityname = Region_Name,
    energydemand_ev = Strombedarf_Personenwagen_BEV_und_PHEV,
    regional_category = Region_Kategorie
  ) %>%
  
  select(
    c(
      "bfs_municipalitynr",
      "year",
      "bfs_municipalityname",
      "energydemand_ev",
      "regional_category"
    )
  )





  
  
## 5) Data wrangling

d.municipalities_solar <- d.municipalities.renamed %>%
  left_join(
    d.solarpotential.renamed %>%
      select(bfs_municipalitynr, solarpotential.electricity, solarpotential.heat),
    by = "bfs_municipalitynr"
  )


##ggplot(d.municipalities_solar, aes(x = bfs_municipalityname, y = solarpotential.electricity))+
  ## geom_point()

## Alternative with tilted names for better readability

ggplot(d.municipalities_solar, aes(x = bfs_municipalityname, y = solarpotential.electricity)) +
  geom_point() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


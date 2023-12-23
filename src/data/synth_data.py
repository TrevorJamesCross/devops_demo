"""
DevOps Demo: Synthesize Data
Author: Trevor Cross
Last Updated: 12/21/23

Create fake time series data for multiple products based on stochastic, deterministic, &
temoral components.
"""

#%% ----------------------
#-- ---Import Libraries---
#-- ----------------------

# import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import datetime libraries
import datetime

# import iterative.ai libraries
from mlem.api import save

# import support libraries
import random as rand
from os.path import expanduser

# set random seed
rand.seed(64)

#%% ------------------------
#-- ---Define Sets of IDs---
#-- ------------------------

# define sets of machine product ids
machine_set = {
    "small_drip_machine",
    "luxury_grinder",
    "large_espresso_machine",
    "compact_cold_brew_maker",
    "premium_single_cup_brewer",
    "automatic_french_press",
    "artisan_pourover_device",
    "travel_size_espresso_maker",
    "commercial_cappuccino_machine",
    "modern_filterless_coffee_maker",
    "sleek_cold_brew_dispenser",
    "retro_espresso_press",
    "personal_pour-over_kit",
    "digital_coffee_grinder",
    "industrial_cold_brew_tower",
    "classic_french_press",
    "high_capacity_drip_brewer",
    "smart_single_serve_machine",
    "vintage_espresso_maker",
    "portable_cold_brew_infuser",
    "gourmet_pourover_station",
    "automatic_grind_and_brew",
    "stylish_cappuccino_system",
    "innovative_espresso_tumbler"
    "chrome_french_press",
    "artisan_cold_brew_dispenser",
    "compact_pourover_kit",
    "premium_coffee_siphon",
    "automatic_milk_frother",
    "sleek_travel_french_press",
    "prosumer_espresso_machine"
}

# define set of tools & accessories product ids
tools_acc_set = {
    "electric_gooseneck_kettle",
    "espresso_stirrer",
    "pourover_server",
    "coffee_scale",
    "milk_frother",
    "coffee_scoop",
    "tamper_tool",
    "thermal_coffee_carafe",
    "coffee_filter_papers",
    "grind_size_adjuster",
    "fancy_coffee_mug",
    "bean_storage_container",
    "drip_tray",
    "espresso_cup_set",
    "coffee_syrup_pump",
    "travel_coffee_tumbler",
    "stainless_steel_straws",
    "pouring_kettle",
    "coffee_grounds_disposal_bin",
    "brewing_timer",
    "manual_coffee_grinder",
    "espresso_portafilter",
    "reusable_coffee_filter",
    "coffee_blend_mixer",
    "adjustable_tamper",
    "latte_art_stencils",
    "coffee_cup_warmer",
    "glass_pourover_carafe",
    "digital_brew_scale",
    "collapsible_coffee_dripper",
    "bamboo_coffee_stirrers",
    "vintage_coffee_spoon",
    "decorative_coffee_tongs",
    "siphon_coffee_maker",
    "copper_frothing_pitcher"
}

# define set of coffee bean ids
bean_set = {
    "colombian_smooth_and_dark",
    "rio_de_abejas",
    "farmhouse_morning_brew",
    "ethiopian_sunrise_blend",
    "hawaiian_paradise_roast",
    "java_jungle_explorer",
    "mystic_mocha_delight",
    "peruvian_harmony",
    "moroccan_spice_infusion",
    "sulawesi_mountain_mist",
    "tahitian_vanilla_sunrise",
    "sumatra_nightshade_reserve",
    "kenyan_safari_blend",
    "guatemalan_honey_harvest",
    "arabian_nights_espresso",
    "brazilian_rainforest_blend",
    "costa_rican_cloud_forest",
    "yirgacheffe_garden_glow",
    "caramel_macchiato_dream",
    "jamaican_blue_mountain",
    "siberian_taiga_blend",
    "madagascan_vanilla_breeze",
    "panamanian_golden_sunrise",
    "sardinian_sea_salt_caramel",
    "italian_amaretto_affair"
}

# define set of apparel ids
apparel_set = {
    "espresso_novelty_shirt",
    "ten_gallon_iced_coffee_hat",
    "mocha_colored_cardigan",
    "latte_art_apron",
    "caffeine_molecule_socks",
    "cappuccino_beanie",
    "coffee_blossom_dress",
    "barista_champion_tie",
    "java_lover_hoodie",
    "cold_brew_canvas_tote"
}

# define dict of product categories
cat_dict = {
    'machines': machine_set,
    'tools_&_accessories': tools_acc_set,
    'beans': bean_set,
    'apparel': apparel_set
}

#%% -----------------------------
#-- ---Define Trend Components---
#-- -----------------------------

# define category parameters
price_ranges_by_cat = {cat: vals for cat, vals in zip([cat_name for cat_name in cat_dict], [(100, 300), (1, 25), (10, 15), (5, 60)])}
base_ranges_by_cat = {cat: vals for cat, vals in zip([cat_name for cat_name in cat_dict], [(10, 20), (15, 40), (40, 90), (10, 30)])}

# define empty objects to hold price info and deterministic components for each prod_id
price_dict = {}
base_comps = []
for cat_name in cat_dict:
    for prod_id in cat_dict[cat_name]:
        price_dict.update({prod_id: rand.uniform(*price_ranges_by_cat[cat_name])})
        base_comps.append(rand.uniform(*base_ranges_by_cat[cat_name]) + 15*np.exp(-10**-4*price_dict[prod_id]**2))

#%% -------------------
#-- ---Generate Data---
#-- -------------------

# define range of dates
date_delta = datetime.timedelta(days=3*365)
date_end = datetime.date.today()
date_start = date_end - date_delta

step_len = datetime.timedelta(days=1)
date_range = [date_start + step_num*step_len for step_num in range(date_delta.days+1)]

# define df contents as dict
list_of_dicts = []
for cat_name in cat_dict:
    for id_num, prod_id in enumerate(cat_dict[cat_name]):

        units_sold = base_comps[id_num]

        for date_num, date in enumerate(date_range):

            if date_num > 0:

                # introduce noise
                units_sold = list_of_dicts[date_num-1]["units_sold"] + rand.gauss(mu=0.5, sigma=2)

                # introduce stochastic temporal correlation
                if cat_name != 'machines' and date_num > 2:
                    temp_corr_coeff = rand.uniform(0.1, 0.3)
                    units_sold = (1-temp_corr_coeff)*units_sold + temp_corr_coeff*sum([list_of_dicts[date_num-n]["units_sold"] for n in range(2)])/3

                # introduce periodicity
                units_sold += 0.05*np.sin(np.pi/6*date.month + np.pi/2)

            list_of_dicts.append({"prod_id": prod_id,
                                  "unit_price": round(price_dict[prod_id], 3),
                                  "prod_category": cat_name,
                                  "sale_date": date,
                                  "units_sold": int(units_sold) if units_sold > 0 else int(rand.choice(range(2)))
                                  })
#%% -----------------
#-- ---Export Data---
#-- -----------------

# convert list_of_dicts to df
df = pd.DataFrame(list_of_dicts)
prods = [rand.choice(list(cat_dict[cat_name])) for cat_name in cat_dict]

# save df using mlem
output_path = f"{expanduser('~')}/projects/devops_demo/data/raw/product_sales_data.csv"
save(df, output_path)

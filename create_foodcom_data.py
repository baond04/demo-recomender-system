"""
create_foodcom_data.py
======================
Tạo bộ dữ liệu mẫu Food.com cho hệ thống GNN+GCL+RAG Ẩm thực.
Sinh ra 3 file: recipes.csv, interactions.csv, tags.csv
"""

import csv, random, os
from pathlib import Path

random.seed(42)

# ============================================================
# DỮ LIỆU MẪU MÓN ĂN (Vietnamese + International)
# ============================================================

RECIPES_RAW = [
    # (id, name, category, cuisine, minutes, calories, tags)
    # ---- Món Việt Nam ----
    (1, "Phở Bò Hà Nội", "soup", "Vietnamese", 180, 450, "beef,noodle,soup,traditional,hanoi"),
    (2, "Phở Gà", "soup", "Vietnamese", 120, 380, "chicken,noodle,soup,traditional"),
    (3, "Bún Bò Huế", "soup", "Vietnamese", 150, 420, "beef,noodle,soup,spicy,hue"),
    (4, "Bún Riêu Cua", "soup", "Vietnamese", 90, 350, "crab,noodle,soup,tomato"),
    (5, "Bánh Mì Thịt Nướng", "sandwich", "Vietnamese", 30, 520, "bread,pork,grilled,street_food"),
    (6, "Bánh Mì Trứng", "sandwich", "Vietnamese", 15, 380, "bread,egg,street_food,fast"),
    (7, "Bánh Mì Pate", "sandwich", "Vietnamese", 10, 430, "bread,pate,street_food"),
    (8, "Cơm Tấm Sườn Bì", "rice", "Vietnamese", 60, 680, "rice,pork,grilled,saigon"),
    (9, "Cơm Rang Dưa Bò", "rice", "Vietnamese", 20, 560, "rice,beef,fried_rice,quick"),
    (10, "Bún Chả Hà Nội", "noodle", "Vietnamese", 45, 490, "pork,noodle,grilled,hanoi"),
    (11, "Gỏi Cuốn Tôm Thịt", "appetizer", "Vietnamese", 30, 220, "shrimp,pork,salad_roll,fresh,healthy"),
    (12, "Chả Giò Hải Sản", "appetizer", "Vietnamese", 40, 350, "seafood,spring_roll,fried,appetizer"),
    (13, "Mì Quảng", "noodle", "Vietnamese", 90, 480, "noodle,pork,shrimp,central_vietnam"),
    (14, "Cao Lầu Hội An", "noodle", "Vietnamese", 90, 460, "noodle,pork,hoian,traditional"),
    (15, "Banh Canh Cua", "soup", "Vietnamese", 60, 390, "crab,thick_noodle,soup"),
    (16, "Chè Khúc Bạch", "dessert", "Vietnamese", 120, 180, "dessert,sweet,cold,almond"),
    (17, "Chè Ba Màu", "dessert", "Vietnamese", 30, 250, "dessert,sweet,cold,colorful"),
    (18, "Bánh Xèo Miền Tây", "pancake", "Vietnamese", 40, 420, "pancake,shrimp,pork,southern"),
    (19, "Lẩu Thái Hải Sản", "hotpot", "Thai-Vietnamese", 50, 380, "seafood,hotpot,spicy,thai"),
    (20, "Lẩu Bò Nhúng Dấm", "hotpot", "Vietnamese", 60, 520, "beef,hotpot,vinegar,southern"),
    (21, "Lẩu Mắm", "hotpot", "Vietnamese", 60, 440, "fish_paste,hotpot,mekong"),
    (22, "Nem Chua Rán", "appetizer", "Vietnamese", 45, 290, "pork,fried,appetizer,fermented"),
    (23, "Bò Lúc Lắc", "main", "Vietnamese", 20, 380, "beef,stir_fry,pepper,quick"),
    (24, "Gà Kho Gừng", "main", "Vietnamese", 35, 350, "chicken,braised,ginger,spicy"),
    (25, "Cá Kho Tộ", "main", "Vietnamese", 60, 320, "fish,braised,caramel,clay_pot"),
    (26, "Thịt Kho Trứng", "main", "Vietnamese", 90, 580, "pork,egg,braised,tet"),
    (27, "Canh Chua Cá", "soup", "Vietnamese", 30, 210, "fish,sour_soup,tamarind,tomato"),
    (28, "Canh Bí Đỏ Tôm", "soup", "Vietnamese", 25, 180, "shrimp,pumpkin,soup,light"),
    (29, "Rau Muống Xào Tỏi", "vegetable", "Vietnamese", 10, 120, "water_spinach,garlic,stir_fry,quick,vegan"),
    (30, "Đậu Phụ Sốt Cà Chua", "vegetarian", "Vietnamese", 20, 160, "tofu,tomato,vegetarian,quick"),
    # ---- Ẩm thực Á Đông ----
    (31, "Sushi Cá Hồi", "japanese", "Japanese", 60, 420, "salmon,sushi,rice,japanese,healthy"),
    (32, "Ramen Chashu", "soup", "Japanese", 240, 680, "pork,ramen,noodle,japanese"),
    (33, "Tempura Hải Sản", "fried", "Japanese", 40, 480, "seafood,tempura,fried,japanese"),
    (34, "Takoyaki Bạch Tuộc", "snack", "Japanese", 30, 320, "octopus,snack,japanese,street"),
    (35, "Okonomiyaki", "pancake", "Japanese", 25, 450, "pancake,japanese,cabbage,pork"),
    (36, "Cơm Chiên Dương Châu", "rice", "Chinese", 15, 520, "rice,fried_rice,egg,chinese,quick"),
    (37, "Vịt Quay Bắc Kinh", "main", "Chinese", 180, 720, "duck,roasted,chinese,beijing"),
    (38, "Dim Sum Há Cảo", "dumpling", "Chinese", 45, 280, "shrimp,dumpling,dim_sum,steamed"),
    (39, "Mì Xào Hải Sản", "noodle", "Chinese", 20, 480, "seafood,noodle,stir_fry,chinese,quick"),
    (40, "Bún Tôm Thái", "salad", "Thai", 30, 260, "shrimp,salad,thai,spicy,lime"),
    (41, "Pad Thai Tôm", "noodle", "Thai", 30, 520, "shrimp,noodle,thai,stir_fry"),
    (42, "Tom Yum Goong", "soup", "Thai", 30, 280, "shrimp,soup,thai,spicy,lemongrass"),
    (43, "Green Curry Gà", "curry", "Thai", 40, 450, "chicken,curry,thai,coconut,spicy"),
    (44, "Cơm Rang Kimchi", "rice", "Korean", 20, 480, "kimchi,rice,fried,korean,quick"),
    (45, "Bibimbap", "rice", "Korean", 40, 520, "rice,vegetables,egg,korean,healthy"),
    (46, "Tteokbokki", "snack", "Korean", 20, 380, "rice_cake,spicy,korean,street"),
    (47, "Korean BBQ Galbi", "bbq", "Korean", 60, 620, "beef,bbq,korean,grilled"),
    (48, "Bulgogi", "main", "Korean", 30, 480, "beef,marinated,korean,sweet"),
    (49, "Mì Udon Nhật", "noodle", "Japanese", 20, 440, "udon,japanese,noodle,soup,light"),
    (50, "Gyoza Chiên", "dumpling", "Japanese", 30, 360, "pork,dumpling,fried,japanese"),
    # ---- Ẩm thực phương Tây ----
    (51, "Bít Tết Bò Sốt Tiêu", "main", "Western", 30, 680, "beef,steak,pepper,western,grilled"),
    (52, "Pizza Margherita", "italian", "Italian", 60, 580, "pizza,cheese,tomato,italian,basil"),
    (53, "Pizza Hải Sản", "italian", "Italian", 60, 620, "pizza,seafood,italian,cheese"),
    (54, "Pasta Carbonara", "pasta", "Italian", 25, 680, "pasta,egg,bacon,italian,cream"),
    (55, "Pasta Bolognese", "pasta", "Italian", 60, 580, "pasta,beef,tomato,italian"),
    (56, "Pasta Aglio e Olio", "pasta", "Italian", 20, 480, "pasta,garlic,olive_oil,italian,quick,vegan"),
    (57, "Risotto Nấm", "rice", "Italian", 40, 520, "risotto,mushroom,italian,creamy"),
    (58, "Burger Bò Phô Mai", "burger", "American", 30, 750, "beef,burger,cheese,american"),
    (59, "Chicken Wings Buffalo", "appetizer", "American", 45, 580, "chicken,wings,spicy,american"),
    (60, "Caesar Salad", "salad", "American", 15, 280, "salad,chicken,healthy,light,quick"),
    (61, "Súp Kem Nấm", "soup", "French", 30, 320, "mushroom,cream_soup,french,creamy"),
    (62, "Quiche Lorraine", "pastry", "French", 60, 480, "egg,bacon,pastry,french"),
    (63, "Ratatouille", "vegetarian", "French", 60, 220, "vegetable,french,vegan,healthy"),
    (64, "Salmon Nướng Sốt Cam", "main", "Western", 30, 420, "salmon,grilled,citrus,healthy"),
    (65, "Tôm Sốt Kem Tỏi", "main", "Western", 20, 380, "shrimp,cream,garlic,quick"),
    # ---- Ăn chay / Healthy ----
    (66, "Salad Quinoa Rau Củ", "salad", "Healthy", 20, 280, "quinoa,salad,healthy,vegan,protein"),
    (67, "Smoothie Bowl Acai", "breakfast", "Healthy", 10, 320, "acai,smoothie,breakfast,healthy,fruit"),
    (68, "Avocado Toast", "breakfast", "Western", 10, 350, "avocado,toast,breakfast,healthy,quick"),
    (69, "Granola Sữa Chua", "breakfast", "Healthy", 5, 380, "granola,yogurt,breakfast,quick,healthy"),
    (70, "Buddha Bowl Chay", "vegetarian", "Healthy", 30, 420, "vegetables,vegan,healthy,bowl,protein"),
    (71, "Đậu Hũ Non Hoa Quả", "dessert", "Vietnamese", 10, 120, "tofu,fruit,dessert,vegan,light"),
    (72, "Súp Rau Củ Hầm", "soup", "Healthy", 60, 180, "vegetable,soup,healthy,vegan,light"),
    (73, "Bánh Yến Mạch Chuối", "breakfast", "Healthy", 15, 280, "oat,banana,breakfast,healthy,quick"),
    (74, "Cơm Gạo Lứt Ức Gà", "main", "Healthy", 40, 380, "brown_rice,chicken,healthy,protein,diet"),
    (75, "Sinh Tố Xanh Cải Bó Xôi", "drink", "Healthy", 5, 120, "spinach,smoothie,green,healthy,vegan"),
    # ---- Đồ uống & Tráng miệng ----
    (76, "Tiramisu", "dessert", "Italian", 30, 480, "dessert,coffee,cream,italian,no_bake"),
    (77, "Creme Brulee", "dessert", "French", 60, 420, "dessert,cream,caramel,french"),
    (78, "Bánh Flan Caramen", "dessert", "Vietnamese", 45, 280, "flan,dessert,caramel,egg,baked"),
    (79, "Panna Cotta Trái Cây", "dessert", "Italian", 20, 320, "dessert,cream,fruit,italian,no_bake"),
    (80, "Kem Mochi Nhân Đậu Đỏ", "dessert", "Japanese", 60, 250, "mochi,ice_cream,japanese,dessert"),
    (81, "Cà Phê Sữa Đá", "drink", "Vietnamese", 5, 180, "coffee,milk,iced,vietnamese,drink"),
    (82, "Trà Sữa Trân Châu", "drink", "Taiwanese", 10, 380, "bubble_tea,milk_tea,taiwanese,drink"),
    (83, "Sinh Tố Bơ", "drink", "Vietnamese", 5, 280, "avocado,smoothie,vietnamese,drink"),
    (84, "Nước Ép Cà Rốt Gừng", "drink", "Healthy", 5, 80, "carrot,ginger,juice,healthy,vegan"),
    (85, "Cocktail Mojito Không Cồn", "drink", "Western", 5, 120, "mocktail,mint,lime,drink,refreshing"),
    # ---- Món ăn đặc biệt ----
    (86, "Hải Sản Nướng Muối Ớt", "seafood", "Vietnamese", 30, 380, "seafood,grilled,spicy,salt,chili"),
    (87, "Cua Rang Me", "seafood", "Vietnamese", 40, 420, "crab,tamarind,spicy,seafood"),
    (88, "Tôm Hùm Nướng Phô Mai", "seafood", "Western", 45, 580, "lobster,cheese,grilled,luxury,seafood"),
    (89, "Mực Nhồi Thịt Hấp", "seafood", "Vietnamese", 40, 320, "squid,stuffed,steamed,seafood"),
    (90, "Nghêu Hấp Xả", "seafood", "Vietnamese", 20, 180, "clam,steamed,lemongrass,quick"),
    (91, "Gà Rang Muối Mặn", "main", "Vietnamese", 35, 400, "chicken,fried,salty,crispy"),
    (92, "Vịt Nấu Chao", "main", "Vietnamese", 60, 520, "duck,fermented_tofu,braised"),
    (93, "Bò Sốt Vang Đỏ", "main", "Vietnamese-French", 180, 580, "beef,red_wine,braised,french"),
    (94, "Sườn Nướng BBQ", "bbq", "Western", 120, 650, "pork_ribs,bbq,grilled,american"),
    (95, "Gà Chiên Nước Mắm", "main", "Vietnamese", 30, 450, "chicken,fried,fish_sauce,sweet"),
    (96, "Thịt Nướng Xiên Que", "bbq", "Vietnamese", 30, 380, "pork,skewer,grilled,street"),
    (97, "Bánh Cuốn Hà Nội", "breakfast", "Vietnamese", 30, 280, "steamed_roll,pork,breakfast,hanoi"),
    (98, "Bánh Chưng Tết", "traditional", "Vietnamese", 480, 420, "sticky_rice,pork,traditional,tet,festive"),
    (99, "Xôi Gấc Đỏ", "traditional", "Vietnamese", 60, 380, "sticky_rice,gac_fruit,traditional,festive"),
    (100, "Bún Ốc Hà Nội", "soup", "Vietnamese", 60, 320, "snail,noodle,soup,hanoi,traditional"),
]

# Thêm 1900 món nữa bằng cách biến thể
EXTRA_CATEGORIES = ["soup", "main", "appetizer", "dessert", "salad", "noodle", "rice", "breakfast", "healthy", "vegetarian"]
EXTRA_CUISINES = ["Vietnamese", "Japanese", "Korean", "Chinese", "Thai", "Italian", "French", "American", "Indian", "Mexican"]
EXTRA_TAGS_POOL = [
    "quick", "easy", "healthy", "spicy", "sweet", "sour", "fried", "grilled", "steamed", "baked",
    "vegan", "vegetarian", "gluten_free", "low_carb", "high_protein", "dairy_free",
    "chicken", "beef", "pork", "seafood", "tofu", "egg", "mushroom", "vegetable",
    "noodle", "rice", "bread", "soup", "salad", "dessert", "snack",
    "family", "party", "romantic", "quick_lunch", "weekend", "tet", "festive"
]

RECIPE_NAME_PARTS = [
    ["Gà", "Bò", "Heo", "Tôm", "Cá", "Rau", "Nấm", "Đậu", "Cua", "Mực"],
    ["Xào", "Kho", "Hấp", "Nướng", "Chiên", "Rang", "Hầm", "Sốt"],
    ["Tỏi", "Ớt", "Gừng", "Sả", "Mè", "Phô Mai", "Nước Mắm", "Tiêu", "Me", "Dừa"]
]


def gen_extra_recipes(start_id, count):
    rows = []
    random.seed(99)
    for i in range(count):
        rid = start_id + i
        part1 = random.choice(RECIPE_NAME_PARTS[0])
        part2 = random.choice(RECIPE_NAME_PARTS[1])
        part3 = random.choice(RECIPE_NAME_PARTS[2])
        name = f"{part1} {part2} {part3}"
        cat = random.choice(EXTRA_CATEGORIES)
        cuisine = random.choice(EXTRA_CUISINES)
        minutes = random.choice([10, 15, 20, 30, 40, 45, 60, 90, 120])
        calories = random.randint(100, 800)
        tags = ",".join(random.sample(EXTRA_TAGS_POOL, k=4))
        rows.append((rid, name, cat, cuisine, minutes, calories, tags))
    return rows


def create_recipes_csv(output_dir):
    path = output_dir / "recipes.csv"
    all_recipes = list(RECIPES_RAW) + gen_extra_recipes(101, 1900)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["recipe_id", "name", "category", "cuisine", "minutes", "calories", "tags"])
        for row in all_recipes:
            writer.writerow(row)
    print(f"✓ Đã tạo recipes.csv ({len(all_recipes)} món)")
    return all_recipes


def create_interactions_csv(output_dir, all_recipes):
    path = output_dir / "interactions.csv"
    recipe_ids = [r[0] for r in all_recipes]
    num_users = 800
    num_interactions = 50000
    random.seed(42)

    rows = []
    import time
    base_ts = int(time.time()) - 365 * 24 * 3600

    for _ in range(num_interactions):
        user_id = random.randint(1, num_users)
        recipe_id = random.choice(recipe_ids)
        rating = random.choice([3.0, 3.5, 4.0, 4.0, 4.5, 4.5, 5.0, 5.0])  # Thiên về positive
        ts = base_ts + random.randint(0, 365 * 24 * 3600)
        rows.append((user_id, recipe_id, rating, ts))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "recipe_id", "rating", "timestamp"])
        for row in rows:
            writer.writerow(row)
    print(f"✓ Đã tạo interactions.csv ({len(rows)} tương tác, {num_users} users)")


def create_tags_csv(output_dir):
    path = output_dir / "tags.csv"
    all_tags = sorted(set(EXTRA_TAGS_POOL + [
        "soup", "main", "appetizer", "dessert", "breakfast",
        "noodle", "rice", "bread", "salad", "hotpot",
        "seafood", "beef", "chicken", "pork", "tofu", "fish",
        "grilled", "fried", "steamed", "baked", "raw", "cold", "hot",
        "vegan", "vegetarian", "spicy", "mild", "sweet", "sour",
        "traditional", "modern", "fusion", "street_food",
        "hanoi", "saigon", "hue", "hoian", "mekong",
        "japanese", "korean", "chinese", "thai", "italian", "french", "american",
        "tet", "party", "family", "romantic", "diet", "healthy"
    ]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag_id", "tag"])
        for i, tag in enumerate(all_tags, 1):
            writer.writerow([i, tag])
    print(f"✓ Đã tạo tags.csv ({len(all_tags)} tags)")


def main():
    # Tạo trong cả 2 vị trí
    dirs = [
        Path("d:/dowload/aidoan/demo-recomender-system/foodcom_data"),
        Path("d:/dowload/aidoan/demo-recomender-system/files_vua_sua/01_RAG_Conversational_System/foodcom_data"),
    ]

    for output_dir in dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Tạo dữ liệu tại: {output_dir}")
        all_recipes = create_recipes_csv(output_dir)
        create_interactions_csv(output_dir, all_recipes)
        create_tags_csv(output_dir)

    print("\n✅ Hoàn thành tạo dữ liệu Food.com!")

if __name__ == "__main__":
    main()

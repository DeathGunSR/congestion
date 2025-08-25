# Context-Aware Network Congestion **Trend** Prediction

This project uses a multi-input LSTM neural network to predict the **trend** of network congestion, rather than its exact value. It is "context-aware," meaning it understands the type of network activity (e.g., web browsing, file transfer) to make smarter, more useful predictions.

The system analyzes the last 5 seconds of traffic to predict whether congestion will worsen in the next second, acting as an early-warning system.

## Project Structure

- `data/`: A directory where you must place your labeled `.pcap` files for training.
- `feature_extractor.py`: A utility module that handles advanced feature engineering.
- `pcap_parser.py`: Processes all `.pcap` files in `data/`, assigns activity labels, and creates a unified dataset.
- `train_model.py`: Trains the LSTM-based **binary classification model**.
- `realtime_predictor.py`: Predicts the congestion trend in real-time and visualizes it with a color-coded chart.
- `requirements.txt`: All Python dependencies.
- `activity_map.json`: Maps activity names to integer IDs.
- `congestion_model.keras`: The saved, trained classification model.
- `scaler.gz`: The saved feature scaler.

## How to Use

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Data Preparation & Training

The model learns to distinguish between normal and problematic traffic patterns for different activities.

1.  **Create a `data` Directory:** If it doesn't exist, create it.
2.  **Add Labeled `.pcap` Files:** Place your `.pcap` files inside the `data/` directory. Name them to reflect the activity.
    -   **Examples:** `web_browsing.pcap`, `gaming_csgo.pcap`, `video_1.pcap`, `file_transfer.pcap`
    -   `gaming_csgo.pcap` and `gaming_fortnite.pcap` will both be labeled as `gaming`.

3.  **Run the Parser:** This script processes all files in `data/` and generates `processed_data.csv` and `activity_map.json`.
    ```bash
    python pcap_parser.py
    ```

4.  **Run the Training Script:** This trains the new classification model.
    ```bash
    python train_model.py
    ```
    The script will output the model's **accuracy** at the end.

### 3. Real-Time Congestion Prediction

The real-time predictor now acts as a warning system.

1.  **Find Your Network Interface:** If needed, run `python realtime_predictor.py --ip YOUR_IP` to see a list of interfaces.

2.  **Run the Predictor with Activity:**
    ```bash
    # Example for predicting 'gaming' traffic
    # Note: Requires admin/sudo privileges
    sudo python realtime_predictor.py --ip 192.168.1.103 --iface wlan0 --activity gaming
    ```
    - Replace the values with your configuration. The activity name must be one of the labels you used for training.

The script will open a live plot showing the actual RTT. The background of the plot will be shaded to indicate the model's prediction:
-   **Green Background:** Network state is predicted to be **Stable/Improving**.
-   **Red Background:** Network state is predicted to be **Worsening**.

---

## تشریح دقیق متدولوژی (رویکرد طبقه‌بندی)

این بخش متدولوژی جدید پروژه را که مبتنی بر طبقه‌بندی (Classification) است، تشریح می‌کند.

### هدف: پیش‌بینی روند ازدحام (Classification Task)

به جای پیش‌بینی مقدار دقیق RTT (یک مسئله رگرسیون)، ما مسئله را به یک **طبقه‌بندی دوتایی (Binary Classification)** تغییر دادیم. هدف جدید مدل، پاسخ به این سوال است: "آیا وضعیت ازدحام در یک ثانیه آینده رو به وخامت خواهد رفت؟"

این رویکرد چندین مزیت دارد:
- **کاربردی‌تر:** به عنوان یک سیستم هشدار، دانستن روند تغییرات مهم‌تر از دانستن مقدار دقیق آن است.
- **ساده‌تر برای مدل:** یادگیری یک مرز تصمیم بین دو کلاس (وخیم شدن / پایدار) برای مدل ساده‌تر از تخمین یک مقدار پیوسته و نویزی است.
- **ارزیابی بهتر:** دقت (Accuracy) مدل به یک معیار قابل فهم برای سنجش عملکرد تبدیل می‌شود.

### مهندسی ویژگی‌های پیشرفته

ما به جای استفاده از داده‌های خام، ویژگی‌های هوشمندانه‌ای را مهندسی می‌کنیم که به مدل درک عمیق‌تری از وضعیت شبکه می‌دهند:
- **آمار RTT:** محاسبه `mean`, `min`, `max`, و `std` (انحراف معیار) RTT در هر بازه زمانی، به مدل اجازه می‌دهد تا پایداری و نوسان تاخیر را درک کند.
- **روند (Momentum):** محاسبه `diff()` یا تغییرات معیارهای کلیدی نسبت به بازه زمانی قبل (مانند `rtt_mean_trend`). این ویژگی به مدل کمک می‌کند تا بفهمد آیا یک معیار در حال افزایش است یا کاهش.
- **پکت‌های گمشده (Packet Loss):** تعداد پکت‌هایی که ارسال شده‌اند اما پاسخ آن‌ها در یک بازه زمانی مشخص (تایم‌اوت) دریافت نشده است. این یک شناسه بسیار قوی و مستقیم برای ازدحام شدید است.

### برچسب‌گذاری داده‌ها (Labeling)

ما یک برچسب دوتایی (`0` یا `1`) برای هر نمونه در دیتاست ایجاد می‌کنیم. این کار در تابع `create_sequences` انجام می‌شود:
- **کلاس ۱ (وخامت):** این برچسب در یکی از دو حالت زیر به یک بازه زمانی اختصاص داده می‌شود:
    1.  **از دست رفتن پکت:** اگر در بازه زمانی آینده، حتی یک پکت گمشده (ارسال شده ولی ACK نشده) وجود داشته باشد.
    2.  **افزایش RTT:** اگر پکت گمشده‌ای وجود نداشته باشد، اما میانگین RTT بیش از یک آستانه مشخص (مثلاً ۲۰٪) افزایش یابد.
- **کلاس ۰ (پایدار):** در غیر این صورت.

این منطق تضمین می‌کند که از دست رفتن پکت، که مهم‌ترین نشانه ازدحام است، همیشه به عنوان یک وضعیت بحرانی شناسایی شود.

### معماری مدل طبقه‌بندی

تغییرات کلیدی زیر در معماری مدل اعمال شده است:
1.  **لایه خروجی:** لایه آخر مدل به `Dense(1, activation='sigmoid')` تغییر یافته است. تابع **sigmoid** یک خروجی بین ۰ و ۱ تولید می‌کند که می‌توان آن را به عنوان "احتمال وخیم شدن وضعیت" تفسیر کرد.
2.  **تابع هزینه (Loss Function):** از `binary_crossentropy` استفاده می‌شود که تابع هزینه استاندارد برای مسائل طبقه‌بندی دوتایی است.
3.  **معیار ارزیابی (Metric):** در کنار Loss، **Accuracy** (دقت) مدل نیز در طول آموزش و تست اندازه‌گیری می‌شود.

### نمایش نتایج در حالت Real-Time

اسکریپت `realtime_predictor.py` خروجی مدل (یک احتمال بین ۰ و ۱) را دریافت کرده و آن را تفسیر می‌کند:
- اگر احتمال خروجی > 0.5 باشد، پیش‌بینی "Worsening" است.
- در غیر این صورت، پیش‌بینی "Stable" است.

این نتیجه به جای یک خط دوم روی نمودار، با **رنگ کردن پس‌زمینه نمودار** نمایش داده می‌شود. این روش نمایش، یک سیستم هشدار بصری بسیار واضح و کاربردی ایجاد می‌کند.

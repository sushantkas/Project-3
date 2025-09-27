import pandas as pd
import numpy as np

class preprocessing:
    """
    A class to preprocess Amazon delivery data for further analysis or modeling.

    Attributes:
        amazon (pd.DataFrame): The preprocessed Amazon delivery DataFrame.

    Methods:
        haversine_vectorized(lat1, lon1, lat2, lon2):
            Computes the Haversine distance between two sets of latitude and longitude coordinates.
    """
    def __init__(self, amazon:pd.DataFrame):
        """
        Initializes the preprocess class with a DataFrame, cleans and preprocesses the data.

        Args:
            amazon (pd.DataFrame): The raw Amazon delivery data.

        Processing steps:
            - Strips whitespace from all object columns.
            - Replaces "NaN" strings with np.nan in object columns.
            - Checks for required columns and their order.
            - Converts latitude and longitude columns to absolute values.
            - Fills missing values in 'Agent_Rating' with the median.
            - Fills missing values in 'Weather', 'Traffic', and 'Order_Time' with their respective modes.
        """
        amazon[amazon.select_dtypes("object").columns] = amazon[amazon.select_dtypes("object").columns].apply(lambda x: x.str.strip())
        amazon['Store_Longitude'] = amazon['Store_Longitude'].abs()
        amazon['Store_Latitude'] = amazon['Store_Latitude'].abs()
        for i in amazon[amazon.select_dtypes("object").columns]:
            amazon[i] = amazon[i].replace("NaN", np.nan)
        _columns = [
            'Order_ID', 'Agent_Age', 'Agent_Rating', 'Store_Latitude',
            'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 'Order_Date',
            'Order_Time', 'Pickup_Time', 'Weather', 'Traffic', 'Vehicle', 'Area',
            'Delivery_Time', 'Category'
        ]
        if all(col in amazon.columns for col in _columns):
            self.amazon = amazon
        else:
            print(f"Mismatch in Columns Please pass all the necessary columns in dataset in Following order: {_columns}")

        # Avoid chained assignment warnings
        self.amazon["Agent_Rating"] = self.amazon["Agent_Rating"].fillna(self.amazon["Agent_Rating"].median())
        self.amazon["Weather"] = self.amazon["Weather"].fillna(self.amazon["Weather"].mode()[0])
        self.amazon["Traffic"] = self.amazon["Traffic"].fillna(self.amazon["Traffic"].mode()[0])
        self.amazon["Order_Time"] = self.amazon["Order_Time"].fillna(self.amazon["Order_Time"].mode()[0])
    

    @staticmethod
    def haversine_vectorized(lat1, lon1, lat2, lon2):
        """
        Computes the Haversine distance between two sets of latitude and longitude coordinates.

        Args:
            lat1 (float, pd.Series, or np.ndarray): Latitude(s) of the first location(s).
            lon1 (float, pd.Series, or np.ndarray): Longitude(s) of the first location(s).
            lat2 (float, pd.Series, or np.ndarray): Latitude(s) of the second location(s).
            lon2 (float, pd.Series, or np.ndarray): Longitude(s) of the second location(s).

        Returns:
            float, pd.Series, or np.ndarray: The distance(s) in kilometers.

        Note:
            The arguments should be in the order:
            'Store_Latitude','Store_Longitude', 'Drop_Latitude', 'Drop_Longitude'
        """
        R = 6371  # Earth radius in km

        # Convert degrees to radians
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        # Differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def get_part_of_day(self):
        """
        Categorizes pickup times into parts of the day.
        
        Returns:
            preprocess: Returns the instance of the class for method chaining.
            
        Raises:
            ValueError: If Pickup_Time contains invalid time formats
        """
        try:
            # Specify format to avoid UserWarning
            hour = pd.to_datetime(self.amazon["Pickup_Time"], format="%H:%M:%S", errors="coerce").dt.hour
            hour = pd.cut(
                hour,
                bins=[0, 5, 12, 17, 21, 24],
                labels=["Night", "Morning", "Afternoon", "Evening", "Night"],
                right=False,
                ordered=False
            )
            self.amazon["Pickup_day_part"] = hour
            return self
        except Exception as e:
            raise ValueError(f"Error processing Pickup_Time: {str(e)}")
        
    def delay(df):
        df["Order_Time"] = pd.to_datetime(df["Order_Time"], format="%H:%M:%S").dt.time
        df["Pickup_Time"] = pd.to_datetime(df["Pickup_Time"], format="%H:%M:%S").dt.time

        order_min = df["Order_Time"].apply(lambda x: x.hour * 60 + x.minute)
        pickup_min = df["Pickup_Time"].apply(lambda x: x.hour * 60 + x.minute)
        
        delays = pickup_min - order_min
        # Fix negative delays (crossing midnight)
        delays = delays.apply(lambda x: x if x >= 0 else x + 24*60)
        
        return delays




def amazon_preprocess(amazon):
    data = preprocessing(amazon)
    data.amazon["Distance_km"] = preprocessing.haversine_vectorized(
        data.amazon['Store_Latitude'],
        data.amazon['Store_Longitude'],
        data.amazon['Drop_Latitude'],
        data.amazon['Drop_Longitude']
    )
    data.amazon["Delay"]=preprocessing.delay(data.amazon)
    data.amazon["Order_day"] = pd.to_datetime(data.amazon['Order_Date']).dt.day_of_week
    data.get_part_of_day()
    data.amazon.set_index("Order_ID", inplace=True)
    return data.amazon
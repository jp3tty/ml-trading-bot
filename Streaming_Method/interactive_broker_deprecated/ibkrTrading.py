import sys
from pathlib import Path

from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.common import TickerId
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from techAnalysis import TechnicalAnalysis

from .orderManager import OrderManager

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

ta = TechnicalAnalysis()
        

class IBConnection(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, wrapper=self)
        self.data = {}
        self.data_ready = False
        self.next_order_id = None


    def connect(self, host, port, clientId):
        super().connect(host, port, clientId)
        thread = threading.Thread(target=self.run)
        thread.start()
        time.sleep(1)
        
    def nextValidId(self, orderID: TickerId):
        super.nextValidId(orderID)
        self.next_order_id = orderID
        return self.next_order_id


    # Call by API, will parse and store data for us
    def historicalData(self, reqId, bar):
        if reqId not in self.data:
            self.data[reqId] = []

        self.data[reqId].append({
            "data": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        self.data_ready = True

    def requestData(self, symbol, duration, barSize):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        # request historical data
        self.reqHistoricalData(
            reqId= 1,                                   # Use a unique reqID, this will be used by later functions
            contract = contract,
            endDateTime= "",                            # Empty string recieves the most recent data
            durationStr=duration,
            barSizeSetting=barSize,
            whatToShow="TRADES",
            useRTH= 0,
            formatDate= 1,
            keepUpToDate= True,                         # Update as we go
            chartOptions=[]                             # no extra options
        )

    def newCandle(self, newBar):
        print(f"New Data Recieved. Latest bar: {newBar}")
        ta = TechnicalAnalysis()
        om = OrderManager(self)

        if(ta.hammerDetect(newBar)):
            print("Hammer Detected, Placing order.")
            
            buyContract = om.create_contract("SPY")
            order = om.create_bracket_order("BUY", 1, newBar["close"], newBar["close"] + 0.4, newBar["close"] - 0.4)
            
            for specificOrder in order:
                om.place_order(buyContract, specificOrder)
            
        

        
    def historicalDataUpdate(self, reqId, bar):
        self.data[reqId].append({
        "data": bar.date,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume
    })
        try:
            if(self.data[1][-1]["date"] is self.data[1][-2]["date"]):
                ta.hammerDetect(self.data[1][-2])
                # do technical analysis
        except Exception as e:
            print("Not enough candles in the dataset to compare.")


def main():
    IBConnect = IBConnection()
    IBConnect.connect("127.0.0.1", 7497, 0)
    # IBConnect.reqMarketDataType(3)                      # Use this line for no paid subscription, so delayed.
    IBConnect.requestData("SPY", "1 D", "1 Min")

    while not(IBConnect.data_ready):
        time.sleep(1)

    df = pd.DataFrame(IBConnect.data)
    print(df)

if __name__ == "__main__":
    main()
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
import os

root=tk.Tk()
root.title("Tamil Nadu Data Analysis Dashboard/தமிழ்நாடு புள்ளி விவரம்")


left_frame=tk.Frame(root,width=300,bg='lightgrey')
left_frame.pack(side='left',fill='y')

right_frame=tk.Frame(root)
right_frame.pack(side='right',expand=True,fill='both')


canvas=tk.Canvas(right_frame)
scrollbar=ttk.Scrollbar(right_frame,orient="vertical",command=canvas.yview)
scrollable_frame=ttk.Frame(canvas)

scrollable_frame.bind("<Configure>",lambda e:canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0),window=scrollable_frame,anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

def display_images(image_paths):
    for widget in scrollable_frame.winfo_children():
        widget.destroy()
    
    for path in image_paths:
        if os.path.exists(path):
            img=Image.open(path)
            img=img.resize((800, 600),Image.Resampling.LANCZOS)
            photo=ImageTk.PhotoImage(img)
            label=tk.Label(scrollable_frame, image=photo)
            label.image=photo
            label.pack(pady=10)
        else:
            label=tk.Label(scrollable_frame, text=f"Image not found: {path}",fg="red")
            label.pack()


image_buttons={
    "Electric Demand":["electricDemand.jpg"],
    "Population":["population.jpg"],
    "Comparison 2015-24":["comparison2015-24.jpg"],
    "Demand Comparison 2015-2024":["demand2015-2024.jpg"],
    "Profit and Loss":["profitAndLoss.jpg"],
    "Temperature Heatmap":["temperatureHeatmap.jpg"],
    "Urban Rural":["urbanRural.jpg"],
    "TemperatureComparison":["TemperatureComparison.jpg"],
    "TemperaturePrediction":["TemperaturePrediction.jpg"],
    "Temperature VS Electricity":["TemperatureVSElectricity.jpg"],
    "Tamil Nadu GDP":["TNGDP.jpg"],
    "GDP Growth":["GDPGrowth.jpg"],
    "GDP with India":["ComparisonWithIndiaGDP.jpg"],
    "GDP with Other States":["ComparisonWithOtherStates.jpg"],
    "QRCode":["qr_code.png"]
}

power_production_images=[f"Power production in {year}-{(year+1)%100}.jpg" for year in range(2015, 2025)]
sector_wise_images=[f"Sector wise electricity consumption in {year}-{(year+1)%100}.jpg" for year in range(2015, 2023)]


row,col=0,0
for name,paths in image_buttons.items():
    btn=tk.Button(left_frame,text=name,width=25,height=2,command=lambda p=paths:display_images(p))
    btn.grid(row=row,column=col,padx=10,pady=5)
    col+=1
    if col>1:
        col=0
        row+=1


tk.Button(left_frame, text="Power Production (2015-2024)", width=25, height=2,command=lambda: display_images(power_production_images)).grid(row=row+1, column=0, padx=10, pady=5)
tk.Button(left_frame, text="Sector-wise Consumption (2015-2023)", width=25, height=2,command=lambda: display_images(sector_wise_images)).grid(row=row+1, column=1, padx=10, pady=5)


tk.Button(left_frame, text="Exit", bg='red', fg='white', width=20, height=2, command=root.destroy).grid(row=row+2, column=0, columnspan=2, pady=20)

root.mainloop()

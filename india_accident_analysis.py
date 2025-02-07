import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import ipywidgets as widgets
from IPython.display import display, clear_output
import streamlit as st
from matplotlib import gridspec
from scipy import stats

def load_and_clean_data():
    # Create DataFrame for 1997-98 data
    data_recent = pd.DataFrame([
        ["Andhra Pradesh", 18836, 19745, 21424],
        ["Arunachal Pradesh", 230, 158, 205],
        ["Assam", 1146, 1187, 1246],
        ["Bihar", 5255, 5344, 7167],
        ["Goa", 1021, 1575, 1313],
        ["Gujarat", 21479, 23746, 25361],
        ["Haryana", 4067, 4575, 4413],
        ["Himachal Pradesh", 1704, 1640, 1639],
        ["Jammu & Kashmir", 3335, 4019, 4242],
        ["Karnataka", 20930, 31302, 30177],
        ["Kerala", 35731, 34656, 32836],
        ["Madhya Pradesh", 23260, 21192, 22608],
        ["Maharashtra", 34426, 34035, 33875],
        ["Manipur", 399, 396, 411],
        ["Meghalaya", 107, 100, 95],
        ["Mizoram", 71, 39, 44],
        ["Nagaland", 132, 87, 47],
        ["Orissa", 5429, 6393, 6777],
        ["Punjab", 1286, 834, 1083],
        ["Rajasthan", 6434, 19946, 20798],
        ["Sikkim", 83, 75, 85],
        ["Tamil Nadu", 42197, 44203, 46723],
        ["Tripura", 562, 582, 297],
        ["Uttar Pradesh", 12737, 13685, 16118],
        ["West Bengal", 8504, 8806, 9195]
    ], columns=["state", "1996", "1997", "1998"])
    
    # Create DataFrame for 1983-86 data
    data_old = pd.DataFrame([
        ["Andhra Pradesh", 9746, 11507, 11306, None],
        ["Assam", 390, 926, 1725, 1469],
        ["Bihar", 7087, 6382, 7192, None],
        ["Gujarat", 13814, 16248, 16514, 17920],
        ["Haryana", 2016, 2203, 2571, 2587],
        ["Himachal Pradesh", 551, 593, 653, 781],
        ["Jammu & Kashmir", 2051, 2027, 2456, 2476],
        ["Karnataka", 13357, 14905, 14700, 14937],
        ["Kerala", 9347, 9324, 10451, None],
        ["Madhya Pradesh", 12024, 15422, 16751, None],
        ["Maharashtra", 44975, 48439, 51635, 53806],
        ["Manipur", 201, 203, 248, 209],
        ["Meghalaya", 291, 418, 389, 237],
        ["Nagaland", 121, 143, 120, 80],
        ["Orissa", 5076, 5231, 5400, 5327],
        ["Punjab", 1071, 991, 1160, 1246],
        ["Rajasthan", 3946, 5623, 5383, 5724],
        ["Sikkim", 41, 67, 74, 95],
        ["Tamil Nadu", 21637, 23393, 24530, 23247],
        ["Tripura", 228, 332, 384, 308],
        ["Uttar Pradesh", 7905, 8790, 13685, 13648],
        ["West Bengal", 11473, 11656, 12357, None]
    ], columns=["state", "1983", "1984", "1985", "1986"])
    
    return data_recent, data_old

def create_choropleth_map(data, year):
    """Create a choropleth map for the selected year"""
    plt.style.use('classic')
    
    # Load India's geographical data
    india = gpd.read_file('Admin2.shp')
    india = india.merge(data, how='left', left_on='ST_NM', right_on='state')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 14), dpi=300, facecolor='white')
    
    # Create choropleth map
    plot = india.plot(column=year,
                     ax=ax,
                     legend=True,
                     legend_kwds={
                         'label': 'Number of Road Accidents',
                         'orientation': 'horizontal',
                         'shrink': 0.5,
                         'fraction': 0.05,
                         'aspect': 25,
                         'format': '%d',
                         'anchor': (0.5, -0.1),
                         'ticks': np.linspace(data[year].min(), data[year].max(), 5)
                     },
                     missing_kwds={'color': '#f5f5f5'},
                     cmap='RdBu_r',
                     edgecolor='black',
                     linewidth=0.5)
    
    # Add state labels with accident numbers
    for idx, row in india.iterrows():
        centroid = row.geometry.centroid
        if not pd.isna(row[year]):
            # Add state name and accident count
            ax.annotate(f"{row['ST_NM']}\n({int(row[year]):,})",
                       xy=(centroid.x, centroid.y),
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=8,
                       fontweight='bold',
                       path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add title
    plt.suptitle('Motor Vehicle Accidents by State', 
                y=0.95,
                fontsize=16,
                fontweight='bold')
    plt.title(f'Year: {year}',
             pad=20,
             fontsize=14)
    
    # Add legend box with color meanings
    legend_text = (
        'Color Guide:\n'
        '■ Dark Red = Highest accidents\n'
        '■ Light Red = High accidents\n'
        '■ Light Blue = Low accidents\n'
        '■ Dark Blue = Lowest accidents\n'
        '■ Grey = Missing data'
    )
    plt.figtext(0.85, 0.3, legend_text, 
                fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig

def create_trend_visualization(data_recent, data_old):
    """Create a visualization showing accident trends over time"""
    plt.style.use('classic')
    
    states_to_show = ['Maharashtra', 'Tamil Nadu', 'Karnataka', 'Gujarat', 'Kerala']
    
    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(states_to_show)))
    years = ['1983', '1984', '1985', '1986'] + ['1996', '1997', '1998']
    
    # Plot data for each state
    for state, color in zip(states_to_show, colors):
        old_values = data_old[data_old['state'] == state][['1983', '1984', '1985', '1986']].values[0]
        recent_values = data_recent[data_recent['state'] == state][['1996', '1997', '1998']].values[0]
        
        # Plot with gap between old and recent data
        ax.plot(['1983', '1984', '1985', '1986'], old_values, 
                marker='o', color=color, label=state, linewidth=2)
        ax.plot(['1996', '1997', '1998'], recent_values, 
                marker='o', color=color, linewidth=2)
        
        # Add dotted line for gap
        ax.plot(['1986', '1996'], [old_values[-1], recent_values[0]], 
                '--', color=color, alpha=0.3)
        
        # Add annotations for significant changes
        if state == 'Maharashtra':  # Example annotation for highest value state
            ax.annotate('Highest recorded\naccidents',
                       xy=('1998', recent_values[-1]),
                       xytext=(10, 30),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                       arrowprops=dict(arrowstyle='->'))
    
    # Add shaded region to highlight data gap
    ax.axvspan('1986', '1996', color='gray', alpha=0.1)
    ax.annotate('No data available\n(1987-1995)',
               xy=('1991', ax.get_ylim()[1]/2),
               ha='center',
               va='center',
               fontsize=10,
               color='gray')
    
    # Customize the plot
    ax.set_title('Road Accident Trends in Major Indian States (1983-1998)', 
                pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Accidents', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend(title='States', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add key insights box
    insights_text = (
        'Key Insights:\n'
        '• Overall increasing trend\n'
        '• Data gap of 10 years\n'
        '• Maharashtra shows\n  highest numbers\n'
        '• Steep rise post-1996'
    )
    plt.figtext(0.15, 0.8, insights_text,
                fontsize=8,
                bbox=dict(facecolor='white', 
                         edgecolor='gray',
                         alpha=0.8))
    
    plt.tight_layout()
    return fig

def visualize_bird_migration(selected_bird=None):
    """Create a visualization of bird migration routes"""
    plt.style.use('classic')
    
    # Create figure with map projection
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    
    try:
        # Read bird migration data
        df = pd.read_csv('bird_migration.csv')
        unique_birds = list(df['bird_name'].unique())
        
        # Simple color scheme
        bird_colors = {
            'All Birds': None,
            **{bird: color for bird, color in zip(
                unique_birds,
                ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(unique_birds)]
            )}
        }
        
        # Filter data based on selection
        if selected_bird and selected_bird != 'All Birds':
            birds_to_plot = [selected_bird]
        else:
            birds_to_plot = unique_birds
        
        # Plot routes
        for bird in birds_to_plot:
            bird_data = df[df['bird_name'] == bird]
            color = bird_colors[bird]
            
            if len(bird_data) < 2:
                continue
            
            # Plot route
            ax.plot(bird_data['longitude'], bird_data['latitude'],
                   marker='o', markersize=4, label=bird, color=color,
                   linewidth=2, alpha=0.8, transform=ccrs.PlateCarree())
            
            # Add start and end points
            ax.annotate('Start',
                       xy=(bird_data['longitude'].iloc[0], 
                           bird_data['latitude'].iloc[0]),
                       xytext=(10, 10), textcoords='offset points',
                       color=color, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.7))
            
            ax.annotate('End',
                       xy=(bird_data['longitude'].iloc[-1], 
                           bird_data['latitude'].iloc[-1]),
                       xytext=(10, 10), textcoords='offset points',
                       color=color, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.7))
        
        # Set map extent
        if not df.empty:
            ax.set_extent([
                df['longitude'].min() - 5,
                df['longitude'].max() + 5,
                df['latitude'].min() - 3,
                df['latitude'].max() + 3
            ])
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add title
        plt.title(f'Bird Migration Routes Across India\n{"All Birds" if not selected_bird or selected_bird == "All Birds" else selected_bird}', 
                 pad=20, fontsize=16, fontweight='bold')
        
        # Add legend
        if len(birds_to_plot) > 0:
            legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                             title='Bird Species', title_fontsize=12)
        
        # Add basic information box
        info_text = (
            'Migration Information:\n'
            '• Dots show stopping points\n'
            '• Lines show flight paths\n'
            f'• Currently showing: {selected_bird or "All Birds"}'
        )
        plt.figtext(0.15, 0.15, info_text,
                   bbox=dict(facecolor='white', alpha=0.7),
                   fontsize=8)
        
    except Exception as e:
        plt.title('Error Loading Bird Migration Data', pad=20, fontsize=16, color='red')
        plt.figtext(0.5, 0.5, f'Error: {str(e)}',
                   ha='center', va='center',
                   color='red', fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    # Load and process data
    data_recent, data_old = load_and_clean_data()
    
    # Set up the Streamlit page
    st.title('India Transportation and Migration Analysis')
    
    # Add tab selection
    tab1, tab2, tab3 = st.tabs(["Accident Distribution", "Accident Trends", "Bird Migration"])
    
    with tab1:
        year = st.selectbox('Select Year', ['1996', '1997', '1998'])
        fig1 = create_choropleth_map(data_recent, year)
        st.pyplot(fig1)
    
    with tab2:
        fig2 = create_trend_visualization(data_recent, data_old)
        st.pyplot(fig2)
    
    with tab3:
        try:
            df = pd.read_csv('bird_migration.csv')
            unique_birds = list(df['bird_name'].unique())
            
            st.write("""
            ### Bird Migration Routes
            Select a specific bird to view its migration route, or view all routes together.
            Each route shows the bird's journey with stopping points along the way.
            """)
            
            # Simple bird selector
            bird_selection = st.selectbox(
                'Select Bird',
                ['All Birds'] + unique_birds
            )
            
            # Create and display bird migration map
            fig3 = visualize_bird_migration(bird_selection)
            st.pyplot(fig3)
            
        except Exception as e:
            st.error(f"Error loading bird migration data: {str(e)}")

# Add helper functions for calculations
def calculate_total_distance(df, bird):
    # Implementation for distance calculation
    return 0.0

def calculate_average_speed(df, bird):
    # Implementation for speed calculation
    return 0.0

def calculate_duration(df, bird):
    # Implementation for duration calculation
    return 0

def calculate_stops(df, bird):
    # Implementation for counting stops
    return 0

# Example of how to fetch data from Movebank (a popular animal tracking database)
def fetch_movebank_data():
    # You'll need to register for an API key
    MOVEBANK_API_KEY = "your_api_key"
    
    # Example birds that are commonly tracked:
    # - White Storks
    # - Bar-headed Geese
    # - Black Kites
    
    return pd.DataFrame({
        'bird_name': [],
        'latitude': [],
        'longitude': [],
        'date': [],
        'altitude': []
    })

if __name__ == "__main__":
    main()

print(plt.style.available)
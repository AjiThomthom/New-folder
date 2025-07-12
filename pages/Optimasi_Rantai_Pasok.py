import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from core.supply_chain_ga import SupplyChainOptimizer

# =====================
# KONFIGURASI UTAMA
# =====================
def main():
    st.title("⚙️ OPTIMASI RANTAI PASOK INDUSTRI")
    st.caption("Model Matematika: Algoritma Genetika Multi-Objective dengan Kendala Kapasitas dan Emisi")
    
    # =====================
    # INPUT PARAMETER
    # =====================
    with st.expander("⚙️ PARAMETER MODEL", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Sumber Daya")
            n_factories = st.slider("Jumlah Pabrik", 2, 10, 3)
            max_capacity = st.number_input("Kapasitas Maks Pabrik (ton/hari)", 100, 5000, 1000)
            production_cost = st.number_input("Biaya Produksi/Unit ($)", 5.0, 20.0, 10.0)
            
        with col2:
            st.subheader("Pasar & Permintaan")
            n_markets = st.slider("Jumlah Pasar", 2, 15, 5)
            min_demand = st.number_input("Permintaan Min Pasar (ton/hari)", 50, 300, 100)
            max_demand = st.number_input("Permintaan Maks Pasar (ton/hari)", 400, 2000, 500)
            
        with col3:
            st.subheader("Kendala Lingkungan")
            transport_rate = st.slider("Biaya Transport ($/km-ton)", 0.1, 2.0, 0.5)
            co2_per_km = st.slider("Emisi CO₂ (kg/km-ton)", 0.1, 1.0, 0.3)
            co2_budget = st.number_input("Batas Emisi CO₂ Harian (kg)", 500, 10000, 2000)
    
    # =====================
    # GENERASI JARINGAN
    # =====================
    if st.button("🧪 Bangun Jaringan Produksi", type="primary"):
        with st.spinner("Membangun model rantai pasok..."):
            # Inisialisasi optimizer
            optimizer = SupplyChainOptimizer(
                n_factories=n_factories,
                n_markets=n_markets,
                production_cost=production_cost,
                transport_rate=transport_rate,
                co2_per_km=co2_per_km,
                max_capacity=max_capacity,
                min_demand=min_demand,
                max_demand=max_demand,
                co2_budget=co2_budget
            )
            
            # Generate data
            model_data = optimizer.generate_network()
            
            # Simpan di session state
            st.session_state['optimizer'] = optimizer
            st.session_state['model_data'] = model_data
            st.session_state['results'] = None
        
        st.success(f"Jaringan dengan {n_factories} pabrik dan {n_markets} pasar berhasil dibuat!")
    
    # =====================
    # VISUALISASI JARINGAN
    # =====================
    if 'model_data' in st.session_state:
        df_nodes = st.session_state['model_data']['nodes']
        df_edges = st.session_state['model_data']['edges']
        
        st.subheader("🌍 Peta Jaringan Produksi")
        fig = go.Figure()
        
        # Tambahkan node pabrik
        factories = df_nodes[df_nodes['type'] == 'factory']
        fig.add_trace(go.Scattergeo(
            lon=factories['lon'],
            lat=factories['lat'],
            text=factories['name'],
            marker=dict(
                size=factories['capacity']/50,
                color='#FF6B00',
                symbol='square'
            ),
            name='Pabrik',
            hoverinfo='text+name'
        ))
        
        # Tambahkan node pasar
        markets = df_nodes[df_nodes['type'] == 'market']
        fig.add_trace(go.Scattergeo(
            lon=markets['lon'],
            lat=markets['lat'],
            text=markets['name'],
            marker=dict(
                size=markets['demand']/20,
                color='#00CC96',
                symbol='circle'
            ),
            name='Pasar',
            hoverinfo='text+name'
        ))
        
        # Tambahkan edge
        for _, edge in df_edges.iterrows():
            fig.add_trace(go.Scattergeo(
                lon=[edge['src_lon'], edge['dest_lon']],
                lat=[edge['src_lat'], edge['dest_lat']],
                mode='lines',
                line=dict(width=0.7, color='#636EFA'),
                opacity=0.5,
                showlegend=False,
                hoverinfo='text',
                text=f"Biaya: ${edge['cost_per_ton']:.2f}/ton<br>Jarak: {edge['distance']:.1f}km"
            ))
        
        fig.update_geos(
            projection_type="natural earth",
            landcolor='#F0F2F6',
            showcountries=True
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # =====================
        # KONTROL OPTIMASI
        # =====================
        st.subheader("⚙️ Kontrol Optimasi")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Parameter Algoritma Genetika**")
            pop_size = st.slider("Ukuran Populasi", 50, 500, 100)
            generations = st.slider("Jumlah Generasi", 50, 1000, 200)
            mutation_rate = st.slider("Laju Mutasi", 0.01, 0.5, 0.1)
            
        with col2:
            st.markdown("**Bobot Prioritas**")
            weight_cost = st.slider("Prioritas Biaya", 0.0, 1.0, 0.7)
            weight_emission = st.slider("Prioritas Emisi", 0.0, 1.0, 0.3)
            
            # Validasi bobot
            if abs(weight_cost + weight_emission - 1.0) > 0.01:
                st.warning("Total bobot harus 1.0. Menyesuaikan otomatis...")
                total = weight_cost + weight_emission
                weight_cost = weight_cost / total
                weight_emission = weight_emission / total
        
        # =====================
        # JALANKAN OPTIMASI
        # =====================
        if st.button("🚀 Jalankan Optimasi", type="primary", use_container_width=True):
            if 'optimizer' not in st.session_state:
                st.error("Bangun jaringan terlebih dahulu!")
                return
                
            with st.spinner("Menjalankan algoritma genetika..."):
                start_time = time.time()
                
                # Jalankan optimasi
                results = st.session_state['optimizer'].solve(
                    population_size=pop_size,
                    generations=generations,
                    mutation_rate=mutation_rate,
                    weights=(weight_cost, weight_emission)
                )
                
                # Simpan hasil
                st.session_state['results'] = results
                st.session_state['run_time'] = time.time() - start_time
                
            st.success(f"Optimasi selesai dalam {st.session_state['run_time']:.2f} detik!")
            st.balloons()
    
    # =====================
    # TAMPILKAN HASIL
    # =====================
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        
        st.subheader("📊 Hasil Optimasi")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Biaya", f"${results['total_cost']:,.2f}")
        col2.metric("Total Emisi CO₂", f"{results['total_emission']:,.2f} kg")
        col3.metric("Utilisasi Rata-rata", f"{results['utilization']:.1%}")
        
        # Tabel alokasi
        st.markdown("**Alokasi Produksi ke Pasar**")
        st.dataframe(results['allocation'].style.format("{:.1f} ton"), use_container_width=True)
        
        # Grafik konvergensi
        st.subheader("📈 Proses Konvergensi Algoritma")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['convergence']['generation'],
            y=results['convergence']['best_cost'],
            mode='lines',
            name='Biaya Terbaik',
            line=dict(color='#FF6B00', width=3)
        )
        fig.add_trace(go.Scatter(
            x=results['convergence']['generation'],
            y=results['convergence']['avg_cost'],
            mode='lines',
            name='Biaya Rata-rata',
            line=dict(color='#636EFA', dash='dash')
        ))
        fig.update_layout(
            xaxis_title="Generasi",
            yaxis_title="Biaya Total ($)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis sensitivitas
        st.subheader("🔍 Analisis Sensitivitas")
        with st.expander("Pengaruh Perubahan Parameter"):
            st.markdown("""
            | Parameter | Dampak pada Biaya | Dampak pada Emisi |
            |-----------|-------------------|-------------------|
            | **+10% Kapasitas Pabrik** | ↓ 5-8% | ↑ 1-3% |
            | **+10% Biaya Transport** | ↑ 8-12% | ↓ 2-5% |
            | **+10% Batas Emisi** | ↓ 3-7% | ↑ 8-15% |
            """)

if __name__ == "__main__":
    main()
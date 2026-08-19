import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { TicketPredictionService, PredictionResponse } from './ticket-prediction.service';
import { NgChartsModule } from 'ng2-charts';
import { ChartConfiguration, ChartType } from 'chart.js';

@Component({
  selector: 'app-ticket-prediction',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, NgChartsModule],
  templateUrl: './ticket-prediction.component.html',
  styleUrls: ['./ticket-prediction.component.css']
})
export class TicketPredictionComponent {
  predictionForm: FormGroup;
  loading = false;
  result: PredictionResponse | null = null;
  error: string | null = null;

  public barChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    indexAxis: 'y',
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Impacto de las variables en la predicción'
      }
    }
  };
  public barChartType: ChartType = 'bar';
  public barChartData: ChartConfiguration['data'] = {
    labels: [],
    datasets: [{ data: [], backgroundColor: [] }]
  };

  constructor(
    private fb: FormBuilder,
    private predictionService: TicketPredictionService
  ) {
    this.predictionForm = this.fb.group({
      Categoria: ['', Validators.required],
      Prioridad: ['', Validators.required],
      Seniority: ['', Validators.required],
      Hora_Creacion: [0, [Validators.required, Validators.min(0), Validators.max(23)]],
      Dia_Semana: ['', Validators.required],
      Tiempo_Resolucion_hrs: [0, [Validators.required, Validators.min(0)]]
    });
  }

  onSubmit(): void {
    if (this.predictionForm.invalid) return;

    this.loading = true;
    this.result = null;
    this.error = null;

    this.predictionService.predictSla(this.predictionForm.value).subscribe({
      next: (res) => {
        this.result = res;
        
        if (res.explicacion_shap && res.explicacion_shap.length > 0) {
          const labels = res.explicacion_shap.map(s => s.feature);
          const data = res.explicacion_shap.map(s => s.impacto);
          const bgColors = data.map(val => val > 0 ? 'rgba(255, 99, 132, 0.7)' : 'rgba(75, 192, 192, 0.7)');

          this.barChartData = {
            labels: labels,
            datasets: [{
              data: data,
              backgroundColor: bgColors,
              label: 'Impacto'
            }]
          };
        }

        this.loading = false;
      },
      error: (err) => {
        this.error = 'Error al comunicarse con la API. Verifica que el backend esté ejecutándose.';
        this.loading = false;
        console.error(err);
      }
    });
  }
}

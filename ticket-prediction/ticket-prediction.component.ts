import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { TicketPredictionService, PredictionResponse } from './ticket-prediction.service';

@Component({
  selector: 'app-ticket-prediction',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './ticket-prediction.component.html',
  styleUrls: ['./ticket-prediction.component.css']
})
export class TicketPredictionComponent {
  predictionForm: FormGroup;
  loading = false;
  result: PredictionResponse | null = null;
  error: string | null = null;

  constructor(
    private fb: FormBuilder,
    private predictionService: TicketPredictionService
  ) {
    this.predictionForm = this.fb.group({
      Categoria: ['', Validators.required],
      Prioridad: ['', Validators.required],
      Seniority: ['', Validators.required],
      Hora_Creacion: [0, [Validators.required, Validators.min(0), Validators.max(23)]]
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

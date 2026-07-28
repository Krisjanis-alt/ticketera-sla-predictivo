import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

/*
 * IMPORTANTE: Configuración de HttpClient
 * 
 * Para que este servicio funcione, debes proveer HttpClient en la configuración de tu app.
 * 
 * Si usas Standalone Components (Angular 15+ con app.config.ts):
 * import { provideHttpClient } from '@angular/common/http';
 * export const appConfig: ApplicationConfig = {
 *   providers: [provideHttpClient(), ...]
 * };
 * 
 * Si usas NgModules (app.module.ts):
 * import { HttpClientModule } from '@angular/common/http';
 * @NgModule({
 *   imports: [HttpClientModule, ...],
 *   ...
 * })
 * export class AppModule { }
 */

export interface TicketInput {
  Categoria: string;
  Prioridad: string;
  Seniority: string;
  Hora_Creacion: number;
}

export interface PredictionResponse {
  riesgo: 'Bajo' | 'Medio' | 'Alto';
  probabilidad: number;
}

@Injectable({
  providedIn: 'root'
})
export class TicketPredictionService {
  private apiUrl = 'http://localhost:8000/predict';

  constructor(private http: HttpClient) {}

  predictSla(ticket: TicketInput): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(this.apiUrl, ticket);
  }
}

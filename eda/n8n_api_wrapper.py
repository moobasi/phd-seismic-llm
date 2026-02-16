"""
n8n API Wrapper for Seismic EDA Automation
==========================================

This provides a REST API that can be called from n8n workflows.

Setup:
------
1. Install dependencies:
   pip install fastapi uvicorn python-multipart

2. Run the server:
   python n8n_api_wrapper.py

3. In n8n, use HTTP Request node to call:
   POST http://localhost:8000/eda/run
   with JSON body containing the config

Alternative: Run as subprocess from n8n Execute Command node:
   python seismic_eda_automation.py "path/to/file.segy" -o output/ --webhook http://n8n:5678/webhook/eda-progress
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("n8n_api")

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")


# Import the EDA module
try:
    from seismic_eda_automation import EDAConfig, SeismicEDAAutomation, EDAResults
    EDA_AVAILABLE = True
except ImportError:
    EDA_AVAILABLE = False
    print("EDA module not found in path")


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Seismic EDA Automation API",
        description="REST API for automated seismic EDA analysis - designed for n8n integration",
        version="5.0"
    )

    # Store running jobs
    jobs: Dict[str, Dict] = {}


    class EDARequest(BaseModel):
        """Request model for EDA analysis"""
        segy_file: str = Field(..., description="Path to SEGY file")
        output_dir: str = Field(default="eda_outputs", description="Output directory")
        webhook_url: Optional[str] = Field(default=None, description="Webhook URL for progress callbacks")
        avg_velocity: float = Field(default=2500.0, description="Average velocity in m/s")
        sample_traces: int = Field(default=30000, description="Number of traces to sample")
        run_statistics: bool = Field(default=True)
        run_quality_checks: bool = Field(default=True)
        run_frequency_analysis: bool = Field(default=True)
        run_resolution_analysis: bool = Field(default=True)
        run_attribute_analysis: bool = Field(default=True)
        run_spatial_analysis: bool = Field(default=True)
        save_figures: bool = Field(default=True)
        enable_cache: bool = Field(default=True)


    class JobStatus(BaseModel):
        """Job status model"""
        job_id: str
        status: str  # pending, running, completed, failed
        progress: float
        phase: str
        started_at: str
        completed_at: Optional[str]
        result_path: Optional[str]
        error: Optional[str]


    def run_eda_task(job_id: str, config: EDAConfig):
        """Background task to run EDA"""
        try:
            jobs[job_id]["status"] = "running"
            jobs[job_id]["phase"] = "Initializing"

            with SeismicEDAAutomation(config) as eda:
                results = eda.run()

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["phase"] = "Complete"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            jobs[job_id]["result_path"] = str(Path(config.output_dir) / "eda_results.json")
            jobs[job_id]["results"] = results.to_dict()

        except Exception as e:
            logger.exception(f"EDA job {job_id} failed")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = datetime.now().isoformat()


    @app.get("/")
    async def root():
        """API root - returns info"""
        return {
            "name": "Seismic EDA Automation API",
            "version": "5.0",
            "endpoints": {
                "POST /eda/run": "Start EDA analysis (async)",
                "POST /eda/run-sync": "Start EDA analysis (sync, waits for completion)",
                "GET /eda/status/{job_id}": "Get job status",
                "GET /eda/result/{job_id}": "Get job results",
                "GET /eda/jobs": "List all jobs",
                "GET /health": "Health check"
            },
            "n8n_usage": {
                "description": "Use HTTP Request node to call these endpoints",
                "example_workflow": "1. HTTP POST to /eda/run-sync with config JSON, 2. Parse response for results"
            }
        }


    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "eda_available": EDA_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }


    @app.post("/eda/run")
    async def run_eda_async(request: EDARequest, background_tasks: BackgroundTasks):
        """
        Start EDA analysis asynchronously

        Returns job_id immediately, use /eda/status/{job_id} to check progress
        """
        if not EDA_AVAILABLE:
            raise HTTPException(status_code=500, detail="EDA module not available")

        # Validate file exists
        if not Path(request.segy_file).exists():
            raise HTTPException(status_code=400, detail=f"SEGY file not found: {request.segy_file}")

        # Create job
        job_id = f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.segy_file) % 10000:04d}"

        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "phase": "Queued",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "result_path": None,
            "error": None,
            "config": request.dict()
        }

        # Create config
        config = EDAConfig(
            segy_file=request.segy_file,
            output_dir=request.output_dir,
            webhook_url=request.webhook_url,
            avg_velocity=request.avg_velocity,
            sample_traces=request.sample_traces,
            run_statistics=request.run_statistics,
            run_quality_checks=request.run_quality_checks,
            run_frequency_analysis=request.run_frequency_analysis,
            run_resolution_analysis=request.run_resolution_analysis,
            run_attribute_analysis=request.run_attribute_analysis,
            run_spatial_analysis=request.run_spatial_analysis,
            save_figures=request.save_figures,
            enable_cache=request.enable_cache
        )

        # Start background task
        background_tasks.add_task(run_eda_task, job_id, config)

        return {
            "job_id": job_id,
            "status": "pending",
            "message": f"EDA job started. Check status at /eda/status/{job_id}"
        }


    @app.post("/eda/run-sync")
    async def run_eda_sync(request: EDARequest):
        """
        Run EDA analysis synchronously (waits for completion)

        Best for n8n workflows that need results immediately.
        Warning: May timeout for large datasets.
        """
        if not EDA_AVAILABLE:
            raise HTTPException(status_code=500, detail="EDA module not available")

        if not Path(request.segy_file).exists():
            raise HTTPException(status_code=400, detail=f"SEGY file not found: {request.segy_file}")

        try:
            config = EDAConfig(
                segy_file=request.segy_file,
                output_dir=request.output_dir,
                webhook_url=request.webhook_url,
                avg_velocity=request.avg_velocity,
                sample_traces=request.sample_traces,
                run_statistics=request.run_statistics,
                run_quality_checks=request.run_quality_checks,
                run_frequency_analysis=request.run_frequency_analysis,
                run_resolution_analysis=request.run_resolution_analysis,
                run_attribute_analysis=request.run_attribute_analysis,
                run_spatial_analysis=request.run_spatial_analysis,
                save_figures=request.save_figures,
                enable_cache=request.enable_cache
            )

            with SeismicEDAAutomation(config) as eda:
                results = eda.run()

            return {
                "status": "completed",
                "processing_time_seconds": results.processing_time_seconds,
                "results": results.to_dict()
            }

        except Exception as e:
            logger.exception("Sync EDA failed")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get("/eda/status/{job_id}")
    async def get_job_status(job_id: str):
        """Get status of an EDA job"""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        job = jobs[job_id]
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "phase": job["phase"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "result_path": job["result_path"],
            "error": job["error"]
        }


    @app.get("/eda/result/{job_id}")
    async def get_job_result(job_id: str):
        """Get results of a completed EDA job"""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        job = jobs[job_id]

        if job["status"] == "running":
            raise HTTPException(status_code=202, detail="Job still running")

        if job["status"] == "failed":
            raise HTTPException(status_code=500, detail=f"Job failed: {job['error']}")

        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")

        return job.get("results", {})


    @app.get("/eda/jobs")
    async def list_jobs():
        """List all jobs"""
        return {
            "total": len(jobs),
            "jobs": [
                {
                    "job_id": j["job_id"],
                    "status": j["status"],
                    "started_at": j["started_at"]
                }
                for j in jobs.values()
            ]
        }


    @app.get("/eda/download/{job_id}")
    async def download_results(job_id: str):
        """Download results JSON file"""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        job = jobs[job_id]
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")

        result_path = job.get("result_path")
        if not result_path or not Path(result_path).exists():
            raise HTTPException(status_code=404, detail="Result file not found")

        return FileResponse(
            result_path,
            media_type="application/json",
            filename=f"eda_results_{job_id}.json"
        )


def main():
    """Run the API server"""
    if not FASTAPI_AVAILABLE:
        print("FastAPI is required. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    print("="*60)
    print("Seismic EDA Automation API Server")
    print("="*60)
    print("\nStarting server at http://0.0.0.0:8000")
    print("\nn8n Integration:")
    print("  Use HTTP Request node with:")
    print("  POST http://localhost:8000/eda/run-sync")
    print("  Content-Type: application/json")
    print("  Body: {\"segy_file\": \"path/to/file.segy\", ...}")
    print("\nSwagger docs: http://localhost:8000/docs")
    print("="*60)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

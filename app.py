
            create_district_pdf(

                villages,

                district_col,

                model_datetime,

                forecast_days,

                district_pdf

            )



            progress.update(

                label="Completed successfully!",

                state="complete"

            )



            st.success(

                f"{forecast_days}-day / "

                f"{forecast_days \* 24}-hour rainfall report is ready."

            )



            st.subheader("📥 Download")



            col1, col2 = st.columns(2)



            with col1:

                with open(warning_pdf, "rb") as f:

                    st.download_button(

                        "📄 Warning PDF",

                        data=f.read(),

                        file_name=warning_pdf.name,

                        mime="application/pdf",

                        use_container_width=True

                    )



            with col2:

                with open(district_pdf, "rb") as f:

                    st.download_button(

                        "📊 District PDF",

                        data=f.read(),

                        file_name=district_pdf.name,

                        mime="application/pdf",

                        use_container_width=True

                    )



            with st.expander("Forecast information"):

                st.write(

                    f"\*\*Forecast:\*\* {forecast_days} days "

                    f"({forecast_days \* 24} hours)"

                )

                st.write(

                    f"\*\*Minimum rainfall:\*\* "

                    f"{float(np.min(rainfall)):.2f} mm"

                )

                st.write(

                    f"\*\*Maximum rainfall:\*\* "

                    f"{float(np.max(rainfall)):.2f} mm"

                )

                st.write(

                    f"\*\*Average grid rainfall:\*\* "

                    f"{float(np.mean(rainfall)):.2f} mm"

                )



        except Exception as e:

            progress.update(

                label="Processing failed",

                state="error"

            )

            st.error(f"Error: {e}")

            st.exception(e)



st.divider()

st.caption(

    "Rainfall values are ECMWF IFS model guidance, not observed rainfall."

)
